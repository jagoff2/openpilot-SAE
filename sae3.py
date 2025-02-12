"""
FrogPilot Sparse Autoencoder for Analyzing ONNX Driving Model with Local Route Logs

This script implements a sparse autoencoder to analyze the ONNX driving model used in FrogPilot.
It processes actual openpilot route logs stored locally to identify specific layers or granular
objects within the model graph that correspond to particular driving actions such as gas and brake
events, lateral actions, desires, etc.

Author: FrogPilot Development Team
Date: 2024-04-27
"""

import os
import re
import bz2
import zstandard as zstd
import capnp
import enum
import sys
import pickle
import warnings
import urllib.parse
from functools import cache, partial
from collections import defaultdict
from itertools import chain
from typing import Callable, Iterable, Iterator, List, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnx
from onnx import numpy_helper
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import multiprocessing
import tqdm

# FrogPilot specific imports
# Assuming the script is placed within FrogPilot's directory structure
# Adjust PYTHONPATH as necessary or ensure this script is within the same directory
from cereal import log as capnp_log
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.api import CommaApi
from openpilot.tools.lib.helpers import RE

# Constants
QLOG_FILENAMES = ['qlog', 'qlog.bz2', 'qlog.zst']
QCAMERA_FILENAMES = ['qcamera.ts']
LOG_FILENAMES = ['rlog', 'rlog.bz2', 'raw_log.bz2', 'rlog.zst']
CAMERA_FILENAMES = ['fcamera.hevc', 'video.hevc']
DCAMERA_FILENAMES = ['dcamera.hevc']
ECAMERA_FILENAMES = ['ecamera.hevc']

DRIVING_MODEL_PATH = "models/supercombo.onnx"  # Local path to the ONNX driving model
SPARSE_COMPONENTS = 50  # Number of sparse components for the autoencoder
SPARSE_ALPHA = 1.0      # Sparsity controlling parameter
ROUTE_ID = "a2a0ccea32023010|2023-07-27--13-01-19"  # Example route ID; replace with actual route ID


# =====================
# Route and LogReader Classes
# =====================

class RouteName:
    def __init__(self, name_str: str):
        self._name_str = name_str
        delim = next((c for c in self._name_str if c in ("|", "/")), "|")
        self._dongle_id, self._time_str = self._name_str.split(delim)
        assert len(self._dongle_id) == 16, self._name_str
        assert len(self._time_str) == 20, self._name_str
        self._canonical_name = f"{self._dongle_id}|{self._time_str}"

    @property
    def canonical_name(self) -> str:
        return self._canonical_name

    @property
    def dongle_id(self) -> str:
        return self._dongle_id

    @property
    def time_str(self) -> str:
        return self._time_str

    def __str__(self) -> str:
        return self._canonical_name


class SegmentName:
    def __init__(self, name_str: str, allow_route_name=False):
        data_dir_path_separator_index = self._find_last_separator(name_str)
        use_data_dir = (data_dir_path_separator_index != -1) and ("|" in name_str)
        self._name_str = name_str[data_dir_path_separator_index + 1:] if use_data_dir else name_str
        self._data_dir = name_str[:data_dir_path_separator_index] if use_data_dir else None
        seg_num_delim = "--" if self._name_str.count("--") == 2 else "/"
        name_parts = self._name_str.rsplit(seg_num_delim, 1)
        if allow_route_name and len(name_parts) == 1:
            name_parts.append("-1")  # no segment number
        self._route_name = RouteName(name_parts[0])
        self._num = int(name_parts[1])
        self._canonical_name = f"{self._route_name.dongle_id}|{self._route_name.time_str}--{self._num}"

    def _find_last_separator(self, name_str):
        split_by_pipe = name_str.rsplit("|", 1)
        if len(split_by_pipe) > 1:
            return name_str.rfind("|")
        split_by_slash = name_str.rsplit("/", 1)
        if len(split_by_slash) > 1:
            return name_str.rfind("/")
        return -1

    @property
    def canonical_name(self) -> str:
        return self._canonical_name

    @property
    def dongle_id(self) -> str:
        return self._route_name.dongle_id

    @property
    def time_str(self) -> str:
        return self._route_name.time_str

    @property
    def segment_num(self) -> int:
        return self._num

    @property
    def route_name(self) -> RouteName:
        return self._route_name

    @property
    def data_dir(self) -> Union[str, None]:
        return self._data_dir

    def __str__(self) -> str:
        return self._canonical_name


class Segment:
    def __init__(self, name: str, log_path: str, qlog_path: str, camera_path: str,
                 dcamera_path: str, ecamera_path: str, qcamera_path: str):
        self._name = SegmentName(name)
        self.log_path = log_path
        self.qlog_path = qlog_path
        self.camera_path = camera_path
        self.dcamera_path = dcamera_path
        self.ecamera_path = ecamera_path
        self.qcamera_path = qcamera_path

    @property
    def name(self):
        return self._name


class ReadMode(enum.Enum):
    RLOG = "r"  # only read rlogs
    QLOG = "q"  # only read qlogs
    SANITIZED = "s"  # read from the commaCarSegments database
    AUTO = "a"  # default to rlogs, fallback to qlogs
    AUTO_INTERACTIVE = "i"  # default to rlogs, fallback to qlogs with a prompt from the user


LogPath = Union[str, None]
ValidFileCallable = Callable[[LogPath], bool]
Source = Callable[['SegmentRange', ReadMode], List[LogPath]]


class LogsUnavailable(Exception):
    pass


class SegmentRange:
    def __init__(self, segment_range: str):
        self.m = re.fullmatch(RE.SEGMENT_RANGE, segment_range)
        assert self.m is not None, f"Segment range is not valid {segment_range}"

    @property
    def route_name(self) -> str:
        return self.m.group("route_name")

    @property
    def dongle_id(self) -> str:
        return self.m.group("dongle_id")

    @property
    def log_id(self) -> str:
        return self.m.group("log_id")

    @property
    def slice(self) -> str:
        return self.m.group("slice") or ""

    @property
    def selector(self) -> Union[str, None]:
        return self.m.group("selector")

    @property
    def seg_idxs(self) -> List[int]:
        m = re.fullmatch(RE.SLICE, self.slice)
        assert m is not None, f"Invalid slice: {self.slice}"
        start, end, step = (None if s is None else int(s) for s in m.groups())
        # one segment specified
        if start is not None and end is None and ':' not in self.slice:
            if start < 0:
                start += get_max_seg_number_cached(self) + 1
            return [start]
        s = slice(start, end, step)
        # no specified end or using relative indexing, need number of segments
        if end is None or end < 0 or (start is not None and start < 0):
            return list(range(get_max_seg_number_cached(self) + 1))[s]
        else:
            return list(range(end + 1))[s]

    def __str__(self) -> str:
        return f"{self.dongle_id}/{self.log_id}" + (f"/{self.slice}" if self.slice else "") + (
            f"/{self.selector}" if self.selector else "")

    def __repr__(self) -> str:
        return self.__str__()


@cache
def get_max_seg_number_cached(sr: 'SegmentRange') -> int:
    try:
        api = CommaApi(get_token())
        max_seg_number = api.get("/v1/route/" + sr.route_name.replace("/", "|"))["maxqlog"]
        assert isinstance(max_seg_number, int)
        return max_seg_number
    except Exception as e:
        raise Exception(
            "unable to get max_segment_number. ensure you have access to this route or the route is public.") from e


class Route:
    def __init__(self, name: str, data_dir: Union[str, None] = None):
        self._name = RouteName(name)
        self.files = None
        if data_dir is not None:
            self._segments = self._get_segments_local(data_dir)
        else:
            self._segments = self._get_segments_remote()
        self.max_seg_number = self._segments[-1].name.segment_num

    @property
    def name(self):
        return self._name

    @property
    def segments(self):
        return self._segments

    def log_paths(self) -> List[LogPath]:
        log_path_by_seg_num = {s.name.segment_num: s.log_path for s in self._segments}
        return [log_path_by_seg_num.get(i, None) for i in range(self.max_seg_number + 1)]

    def qlog_paths(self) -> List[LogPath]:
        qlog_path_by_seg_num = {s.name.segment_num: s.qlog_path for s in self._segments}
        return [qlog_path_by_seg_num.get(i, None) for i in range(self.max_seg_number + 1)]

    def camera_paths(self) -> List[LogPath]:
        camera_path_by_seg_num = {s.name.segment_num: s.camera_path for s in self._segments}
        return [camera_path_by_seg_num.get(i, None) for i in range(self.max_seg_number + 1)]

    def dcamera_paths(self) -> List[LogPath]:
        dcamera_path_by_seg_num = {s.name.segment_num: s.dcamera_path for s in self._segments}
        return [dcamera_path_by_seg_num.get(i, None) for i in range(self.max_seg_number + 1)]

    def ecamera_paths(self) -> List[LogPath]:
        ecamera_path_by_seg_num = {s.name.segment_num: s.ecamera_path for s in self._segments}
        return [ecamera_path_by_seg_num.get(i, None) for i in range(self.max_seg_number + 1)]

    def qcamera_paths(self) -> List[LogPath]:
        qcamera_path_by_seg_num = {s.name.segment_num: s.qcamera_path for s in self._segments}
        return [qcamera_path_by_seg_num.get(i, None) for i in range(self.max_seg_number + 1)]

    def _get_segments_remote(self) -> List[Segment]:
        api = CommaApi(get_token())
        route_files = api.get('v1/route/' + self.name.canonical_name + '/files')
        self.files = list(chain.from_iterable(route_files.values()))
        segments = {}
        for url in self.files:
            _, dongle_id, time_str, segment_num, fn = urlparse(url).path.rsplit('/', maxsplit=4)
            segment_name = f'{dongle_id}|{time_str}--{segment_num}'
            if segments.get(segment_name):
                segments[segment_name] = Segment(
                    segment_name,
                    url if fn in LOG_FILENAMES else segments[segment_name].log_path,
                    url if fn in QLOG_FILENAMES else segments[segment_name].qlog_path,
                    url if fn in CAMERA_FILENAMES else segments[segment_name].camera_path,
                    url if fn in DCAMERA_FILENAMES else segments[segment_name].dcamera_path,
                    url if fn in ECAMERA_FILENAMES else segments[segment_name].ecamera_path,
                    url if fn in QCAMERA_FILENAMES else segments[segment_name].qcamera_path,
                )
            else:
                segments[segment_name] = Segment(
                    segment_name,
                    url if fn in LOG_FILENAMES else None,
                    url if fn in QLOG_FILENAMES else None,
                    url if fn in CAMERA_FILENAMES else None,
                    url if fn in DCAMERA_FILENAMES else None,
                    url if fn in ECAMERA_FILENAMES else None,
                    url if fn in QCAMERA_FILENAMES else None,
                )
        return sorted(segments.values(), key=lambda seg: seg.name.segment_num)

    def _get_segments_local(self, data_dir: str) -> List[Segment]:
        files = os.listdir(data_dir)
        segment_files = defaultdict(list)
        for f in files:
            fullpath = os.path.join(data_dir, f)
            explorer_match = re.match(RE.EXPLORER_FILE, f)
            op_match = re.match(RE.OP_SEGMENT_DIR, f)
            if explorer_match:
                segment_name = explorer_match.group('segment_name')
                fn = explorer_match.group('file_name')
                if segment_name.replace('_', '|').startswith(self.name.canonical_name):
                    segment_files[segment_name].append((fullpath, fn))
            elif op_match and os.path.isdir(fullpath):
                segment_name = op_match.group('segment_name')
                if segment_name.startswith(self.name.canonical_name):
                    for seg_f in os.listdir(fullpath):
                        segment_files[segment_name].append((os.path.join(fullpath, seg_f), seg_f))
            elif f == self.name.canonical_name:
                for seg_num in os.listdir(fullpath):
                    if not seg_num.isdigit():
                        continue
                    segment_name = f'{self.name.canonical_name}--{seg_num}'
                    for seg_f in os.listdir(os.path.join(fullpath, seg_num)):
                        segment_files[segment_name].append((os.path.join(fullpath, seg_num, seg_f), seg_f))
        segments = []
        for segment, files in segment_files.items():
            try:
                log_path = next(path for path, filename in files if filename in LOG_FILENAMES)
            except StopIteration:
                log_path = None
            try:
                qlog_path = next(path for path, filename in files if filename in QLOG_FILENAMES)
            except StopIteration:
                qlog_path = None
            try:
                camera_path = next(path for path, filename in files if filename in CAMERA_FILENAMES)
            except StopIteration:
                camera_path = None
            try:
                dcamera_path = next(path for path, filename in files if filename in DCAMERA_FILENAMES)
            except StopIteration:
                dcamera_path = None
            try:
                ecamera_path = next(path for path, filename in files if filename in ECAMERA_FILENAMES)
            except StopIteration:
                ecamera_path = None
            try:
                qcamera_path = next(path for path, filename in files if filename in QCAMERA_FILENAMES)
            except StopIteration:
                qcamera_path = None
            segments.append(Segment(segment, log_path, qlog_path, camera_path, dcamera_path, ecamera_path, qcamera_path))
        if len(segments) == 0:
            raise ValueError(f'Could not find segments for route {self.name.canonical_name} in data directory {data_dir}')
        return sorted(segments, key=lambda seg: seg.name.segment_num)


class _LogFileReader:
    def __init__(self, fn: str, canonicalize: bool = True, only_union_types: bool = False, sort_by_time: bool = False,
                 dat: bytes = None):
        self.data_version = None
        self._only_union_types = only_union_types

        ext = None
        if not dat:
            _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)
            if ext not in ('', '.bz2', '.zst'):
                # old rlogs weren't compressed
                raise ValueError(f"unknown extension {ext}")
            with FileReader(fn) as f:
                dat = f.read()

        # Decompress data based on extension
        if ext == ".bz2" or dat.startswith(b'BZh9'):
            dat = bz2.decompress(dat)
        elif ext == ".zst" or dat.startswith(b'\x28\xB5\x2F\xFD'):
            # Zstandard frame header
            dat = zstd.decompress(dat)

        ents = capnp_log.Event.read_multiple_bytes(dat)
        self._ents = []
        try:
            for e in ents:
                self._ents.append(e)
        except capnp.KjException:
            warnings.warn("Corrupted events detected", RuntimeWarning, stacklevel=1)
        if sort_by_time:
            self._ents.sort(key=lambda x: x.logMonoTime)

    def __iter__(self) -> Iterator[capnp._DynamicStructReader]:
        for ent in self._ents:
            if self._only_union_types:
                try:
                    ent.which()
                    yield ent
                except capnp.lib.capnp.KjException:
                    pass
            else:
                yield ent


@cache
def default_valid_file(fn: LogPath) -> bool:
    return fn is not None and os.path.exists(fn)


def auto_strategy(rlog_paths: List[LogPath], qlog_paths: List[LogPath], interactive: bool,
                  valid_file: ValidFileCallable) -> List[LogPath]:
    """
    Auto-select logs based on availability.
    
    Args:
        rlog_paths (List[LogPath]): List of rlog file paths.
        qlog_paths (List[LogPath]): List of qlog file paths.
        interactive (bool): Whether to prompt the user for fallback.
        valid_file (ValidFileCallable): Function to validate file existence.
    
    Returns:
        List[LogPath]: Selected log paths.
    """
    missing_rlogs = sum(1 for rlog in rlog_paths if rlog is None or not valid_file(rlog))
    if missing_rlogs != 0:
        if interactive:
            user_input = input(f"{missing_rlogs}/{len(rlog_paths)} rlogs were not found, would you like to fallback to qlogs for those segments? (y/n) ")
            if user_input.lower() != "y":
                return rlog_paths
        else:
            print(f"{missing_rlogs}/{len(rlog_paths)} rlogs were not found, falling back to qlogs for those segments...")
        return [rlog if valid_file(rlog) else (qlog if valid_file(qlog) else None)
                for (rlog, qlog) in zip(rlog_paths, qlog_paths)]
    return rlog_paths


def apply_strategy(mode: ReadMode, rlog_paths: List[LogPath], qlog_paths: List[LogPath],
                   valid_file: ValidFileCallable = default_valid_file) -> List[LogPath]:
    """
    Apply the selected read strategy to determine which logs to load.
    
    Args:
        mode (ReadMode): Mode indicating how to select logs.
        rlog_paths (List[LogPath]): List of rlog file paths.
        qlog_paths (List[LogPath]): List of qlog file paths.
        valid_file (ValidFileCallable): Function to validate file existence.
    
    Returns:
        List[LogPath]: Selected log paths.
    """
    if mode == ReadMode.RLOG:
        return rlog_paths
    elif mode == ReadMode.QLOG:
        return qlog_paths
    elif mode == ReadMode.AUTO:
        return auto_strategy(rlog_paths, qlog_paths, False, valid_file)
    elif mode == ReadMode.AUTO_INTERACTIVE:
        return auto_strategy(rlog_paths, qlog_paths, True, valid_file)
    raise ValueError(f"invalid mode: {mode}")


def comma_api_source(sr: SegmentRange, mode: ReadMode) -> List[LogPath]:
    """
    Source logs from Comma API.
    
    Args:
        sr (SegmentRange): The segment range.
        mode (ReadMode): Mode indicating which logs to read.
    
    Returns:
        List[LogPath]: List of log paths.
    """
    route = Route(sr.route_name)
    rlog_paths = route.log_paths()
    qlog_paths = route.qlog_paths()

    def valid_file(fn):
        return fn is not None

    return apply_strategy(mode, rlog_paths, qlog_paths, valid_file=valid_file)


def internal_source(sr: SegmentRange, mode: ReadMode, file_ext: str = "bz2") -> List[LogPath]:
    """
    Source logs from internal storage.
    
    Args:
        sr (SegmentRange): The segment range.
        mode (ReadMode): Mode indicating which logs to read.
        file_ext (str): File extension to look for.
    
    Returns:
        List[LogPath]: List of log paths.
    """
    # Placeholder for internal_source_available function
    def internal_source_available_mock():
        # In production, implement actual check
        return True

    if not internal_source_available_mock():
        raise LogsUnavailable("Internal source not available")

    def get_internal_url(sr: SegmentRange, seg: int, file: str) -> str:
        return f"cd:/{sr.dongle_id}/{sr.log_id}/{seg}/{file}.{file_ext}"

    rlog_paths = [get_internal_url(sr, seg, "rlog") for seg in sr.seg_idxs]
    qlog_paths = [get_internal_url(sr, seg, "qlog") for seg in sr.seg_idxs]
    return apply_strategy(mode, rlog_paths, qlog_paths)


def openpilotci_source(sr: SegmentRange, mode: ReadMode, file_ext: str = "bz2") -> List[LogPath]:
    """
    Source logs from OpenPilot CI.
    
    Args:
        sr (SegmentRange): The segment range.
        mode (ReadMode): Mode indicating which logs to read.
        file_ext (str): File extension to look for.
    
    Returns:
        List[LogPath]: List of log paths.
    """
    def get_url_mock(route_name: str, seg: int, file: str) -> str:
        # In production, implement actual URL retrieval
        return f"/data/logs/{route_name}/{seg}/{file}.{file_ext}"

    rlog_paths = [get_url_mock(sr.route_name, seg, f"rlog.{file_ext}") for seg in sr.seg_idxs]
    qlog_paths = [get_url_mock(sr.route_name, seg, f"qlog.{file_ext}") for seg in sr.seg_idxs]
    return apply_strategy(mode, rlog_paths, qlog_paths)


def comma_car_segments_source(sr: SegmentRange, mode: ReadMode) -> List[LogPath]:
    """
    Source logs from commaCarSegments.
    
    Args:
        sr (SegmentRange): The segment range.
        mode (ReadMode): Mode indicating which logs to read.
    
    Returns:
        List[LogPath]: List of log paths.
    """
    def get_comma_segments_url_mock(route_name: str, seg: int) -> str:
        # In production, implement actual URL retrieval
        return f"https://comma.com/segments/{route_name}/{seg}/rlog.bz2"

    return [get_comma_segments_url_mock(sr.route_name, seg) for seg in sr.seg_idxs]


def parse_indirect(identifier: str) -> str:
    """
    Parse indirect identifiers to extract the relevant part.
    
    Args:
        identifier (str): The identifier string.
    
    Returns:
        str: Parsed identifier.
    """
    if "useradmin.comma.ai" in identifier:
        query = parse_qs(urlparse(identifier).query)
        return query["onebox"][0]
    return identifier


def parse_direct(identifier: str) -> Union[str, None]:
    """
    Parse direct identifiers to determine if they are URLs or file paths.
    
    Args:
        identifier (str): The identifier string.
    
    Returns:
        Union[str, None]: Parsed identifier or None if invalid.
    """
    if identifier.startswith(("http://", "https://", "cd:/")) or os.path.exists(identifier):
        return identifier
    return None


def comma_api_source(sr: SegmentRange, mode: ReadMode) -> List[LogPath]:
    return comma_api_source(sr, mode)


def internal_source_mock(sr: SegmentRange, mode: ReadMode, file_ext: str = "bz2") -> List[LogPath]:
    return internal_source(sr, mode, file_ext)


def openpilotci_source_mock(sr: SegmentRange, mode: ReadMode, file_ext: str = "bz2") -> List[LogPath]:
    return openpilotci_source(sr, mode, file_ext)


def comma_car_segments_source_mock(sr: SegmentRange, mode: ReadMode) -> List[LogPath]:
    return comma_car_segments_source(sr, mode)


def auto_source(sr: SegmentRange, mode: ReadMode, sources: List[Source] = None) -> List[LogPath]:
    """
    Automatically select the best source to retrieve logs.
    
    Args:
        sr (SegmentRange): The segment range.
        mode (ReadMode): Mode indicating which logs to read.
        sources (List[Source]): List of source functions.
    
    Returns:
        List[LogPath]: Selected log paths.
    """
    if mode == ReadMode.SANITIZED:
        return comma_car_segments_source_mock(sr, mode)
    if sources is None:
        sources = [internal_source_mock, openpilotci_source_mock,
                   comma_api_source, comma_car_segments_source_mock]
    exceptions = {}
    # for automatic fallback modes, auto_source needs to first check if rlogs exist for any source
    if mode in [ReadMode.AUTO, ReadMode.AUTO_INTERACTIVE]:
        for source in sources:
            try:
                return check_source(source, sr, ReadMode.RLOG)
            except Exception:
                pass
    # Automatically determine viable source
    for source in sources:
        try:
            return check_source(source, sr, mode)
        except Exception as e:
            exceptions[source.__name__] = e
    raise LogsUnavailable("auto_source could not find any valid source, exceptions for sources:\n  - " +
                          "\n  - ".join([f"{k}: {repr(v)}" for k, v in exceptions.items()]))


def direct_source(file_or_url: str) -> List[LogPath]:
    """
    Source logs directly from a given file path or URL.
    
    Args:
        file_or_url (str): File path or URL.
    
    Returns:
        List[LogPath]: List containing the single file or URL.
    """
    return [file_or_url]


def get_invalid_files(files: Iterable[LogPath]) -> Iterable[LogPath]:
    """
    Generator to yield invalid files.
    
    Args:
        files (Iterable[LogPath]): Iterable of file paths.
    
    Yields:
        LogPath: Invalid file paths.
    """
    for f in files:
        if f is None or not os.path.exists(f):
            yield f


def check_source(source: Source, sr: SegmentRange, mode: ReadMode) -> List[LogPath]:
    """
    Check and validate the source for log retrieval.
    
    Args:
        source (Source): Source function.
        sr (SegmentRange): The segment range.
        mode (ReadMode): Mode indicating which logs to read.
    
    Returns:
        List[LogPath]: Validated log paths.
    """
    files = source(sr, mode)
    assert len(files) > 0, "No files on source"
    assert next(get_invalid_files(files), False) is False, "Some files are invalid"
    return files


class LogsUnavailable(Exception):
    pass


class RouteNameError(Exception):
    pass


class SegmentRangeError(Exception):
    pass


class LogReader:
    def __init__(self, identifier: Union[str, List[str]], default_mode: ReadMode = ReadMode.RLOG,
                 source: Source = auto_source, sort_by_time: bool = False, only_union_types: bool = False):
        """
        Initialize the LogReader with the given identifier(s).
        
        Args:
            identifier (Union[str, List[str]]): Route identifier(s).
            default_mode (ReadMode): Default mode for reading logs.
            source (Source): Source function for log retrieval.
            sort_by_time (bool): Whether to sort logs by time.
            only_union_types (bool): Whether to filter only union types.
        """
        self.default_mode = default_mode
        self.source = source
        self.identifier = identifier if isinstance(identifier, list) else [identifier]
        self.sort_by_time = sort_by_time
        self.only_union_types = only_union_types
        self.__lrs: dict[int, _LogFileReader] = {}
        self.reset()

    def _parse_identifier(self, identifier: str) -> List[LogPath]:
        """
        Parse the given identifier to retrieve log paths.
        
        Args:
            identifier (str): Route or log identifier.
        
        Returns:
            List[LogPath]: List of log paths.
        """
        # Parse indirect identifiers (e.g., URLs with query parameters)
        identifier = parse_indirect(identifier)
        # Parse direct identifiers (file paths or URLs)
        direct_parsed = parse_direct(identifier)
        if direct_parsed is not None:
            return direct_source(identifier)
        # Otherwise, treat as SegmentRange
        sr = SegmentRange(identifier)
        mode = self.default_mode if sr.selector is None else ReadMode(sr.selector)
        identifiers = self.source(sr, mode)
        invalid_count = len(list(get_invalid_files(identifiers)))
        assert invalid_count == 0, (f"{invalid_count}/{len(identifiers)} invalid log(s) found, please ensure all logs " +
                                    "are uploaded or auto fallback to qlogs with '/a' selector at the end of the route name.")
        return identifiers

    def _get_lr(self, i: int) -> _LogFileReader:
        """
        Retrieve the LogFileReader for the given index.
        
        Args:
            i (int): Index of the log.
        
        Returns:
            _LogFileReader: Log file reader instance.
        """
        if i not in self.__lrs:
            self.__lrs[i] = _LogFileReader(self.logreader_identifiers[i],
                                          sort_by_time=self.sort_by_time,
                                          only_union_types=self.only_union_types)
        return self.__lrs[i]

    def __iter__(self) -> Iterator[capnp._DynamicStructReader]:
        """
        Iterator to yield log messages.
        
        Yields:
            capnp._DynamicStructReader: Log messages.
        """
        for i in range(len(self.logreader_identifiers)):
            yield from self._get_lr(i)

    def _run_on_segment(self, func: Callable, i: int):
        """
        Run a function on a specific log segment.
        
        Args:
            func (Callable): Function to run.
            i (int): Index of the log.
        
        Returns:
            Any: Result of the function.
        """
        return func(self._get_lr(i))

    def run_across_segments(self, num_processes: int, func: Callable, desc: str = None):
        """
        Run a function across all log segments using multiprocessing.
        
        Args:
            num_processes (int): Number of parallel processes.
            func (Callable): Function to execute on each segment.
            desc (str, optional): Description for the progress bar.
        
        Returns:
            list: Aggregated results from all segments.
        """
        with multiprocessing.Pool(num_processes) as pool:
            ret = []
            num_segs = len(self.logreader_identifiers)
            for p in tqdm.tqdm(pool.imap(partial(self._run_on_segment, func), range(num_segs)),
                              total=num_segs, desc=desc):
                ret.extend(p)
            return ret

    def reset(self):
        """
        Reset the LogReader and parse all identifiers to retrieve log paths.
        """
        self.logreader_identifiers = []
        for identifier in self.identifier:
            self.logreader_identifiers.extend(self._parse_identifier(identifier))

    @staticmethod
    def from_bytes(dat: bytes):
        """
        Create a LogReader from raw bytes.
        
        Args:
            dat (bytes): Raw log data.
        
        Returns:
            _LogFileReader: Log file reader instance.
        """
        return _LogFileReader("", dat=dat)

    def filter(self, msg_type: str):
        """
        Generator to filter messages by type.
        
        Args:
            msg_type (str): Type of message to filter.
        
        Yields:
            capnp._DynamicStructReader: Filtered messages.
        """
        return (getattr(m, m.which()) for m in filter(lambda m: m.which() == msg_type, self))

    def first(self, msg_type: str):
        """
        Retrieve the first message of the specified type.
        
        Args:
            msg_type (str): Type of message to retrieve.
        
        Returns:
            capnp._DynamicStructReader or None: The first matching message or None.
        """
        return next(self.filter(msg_type), None)


# =====================
# Sparse Autoencoder Implementation
# =====================

def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """
    Load the ONNX model from the local path.

    Args:
        model_path (str): Local file path to the ONNX model.

    Returns:
        onnx.ModelProto: Loaded ONNX model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    print(f"Loading ONNX model from {model_path}...")
    model = onnx.load(model_path)
    print("Model loaded successfully.")
    return model


def extract_layer_names(model: onnx.ModelProto) -> list:
    """
    Extract all layer/output names from the ONNX model.

    Args:
        model (onnx.ModelProto): The loaded ONNX model.

    Returns:
        list: List of layer/output names.
    """
    layer_names = []
    for node in model.graph.node:
        for output in node.output:
            layer_names.append(output)
    print(f"Extracted {len(layer_names)} layer/output names from the model.")
    return layer_names


def load_route_logs(route_id: str) -> List[str]:
    """
    Load the paths of rlog.bz2 and rlog.zst files for the given route.

    Args:
        route_id (str): Identifier for the route.

    Returns:
        List[str]: List of paths to compressed rlog files.
    """
    route = Route(route_id)
    log_paths = route.log_paths()
    if not log_paths:
        raise FileNotFoundError(f"No log files found for route ID: {route_id}")
    print(f"Found {len(log_paths)} log files for route {route_id}.")
    return log_paths


def extract_features_from_logs(log_paths: List[str]) -> dict:
    """
    Extract relevant features from the rlog files.

    Args:
        log_paths (List[str]): List of paths to rlog files.

    Returns:
        dict: Dictionary mapping feature names to their extracted numpy arrays.
    """
    layer_features = {
        'vEgo': [],
        'steeringAngleDeg': [],
        'gasPressed': [],
        'brakePressed': [],
        'steeringPressed': [],
        'desiredCruiseSpeed': [],
        'desiredLongControlState': []
        # Add more features as needed
    }

    print("Extracting features from logs...")
    for log_path in log_paths:
        if log_path is None:
            continue
        log_reader = LogReader(log_path, sort_by_time=True)
        for msg in log_reader:
            # Extract 'carState' messages
            if msg.which() == 'carState':
                car_state = msg.carState
                layer_features['vEgo'].append(car_state.vEgo)  # Vehicle speed
                layer_features['steeringAngleDeg'].append(car_state.steeringAngleDeg)  # Steering angle
                layer_features['steeringPressed'].append(car_state.steeringPressed)  # Steering wheel pressed

            # Extract 'controlsState' messages
            elif msg.which() == 'controlsState':
                controls_state = msg.controlsState
                layer_features['gasPressed'].append(controls_state.gasPressed)  # Gas pedal pressed
                layer_features['brakePressed'].append(controls_state.brakePressed)  # Brake pedal pressed
                layer_features['desiredCruiseSpeed'].append(controls_state.desiredCruiseSpeed)  # Desired cruise speed
                layer_features['desiredLongControlState'].append(controls_state.desiredLongControlState)  # Desired longitudinal control state

            # Extract 'driverMonitoringState' messages for desires (example)
            elif msg.which() == 'driverMonitoringState':
                desires = msg.driverMonitoringState.desire
                layer_features['desiredLongControlState'].append(desires.longitudinal)  # Desired longitudinal action
                # Add more desires as needed

            # Continue extracting other relevant messages and features

    # Convert lists to numpy arrays and handle missing data
    for feature in layer_features:
        if layer_features[feature]:
            layer_features[feature] = np.array(layer_features[feature])
        else:
            layer_features[feature] = np.array([])  # Handle empty features

    print("Feature extraction completed.")
    return layer_features


def prepare_data_for_autoencoder(layer_features: dict) -> (np.ndarray, list):
    """
    Normalize, flatten, and concatenate layer features to form the dataset for the autoencoder.

    Args:
        layer_features (dict): Extracted layer features.

    Returns:
        tuple: Prepared dataset as a numpy array and list of feature names.
    """
    data_list = []
    feature_names = []
    for feature, data in layer_features.items():
        if data.size == 0:
            continue  # Skip features with no data
        # Normalize features to have zero mean and unit variance
        data_normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
        data_list.append(data_normalized.reshape(-1, 1))  # Reshape for concatenation
        feature_names.append(feature)

    if not data_list:
        raise ValueError("No features extracted from logs.")

    dataset = np.hstack(data_list)
    print(f"Prepared dataset with shape {dataset.shape}.")
    return dataset, feature_names


def train_sparse_autoencoder(data: np.ndarray, n_components: int, alpha: float) -> SparsePCA:
    """
    Train a Sparse PCA autoencoder on the dataset.

    Args:
        data (np.ndarray): The input data.
        n_components (int): Number of sparse components.
        alpha (float): Sparsity controlling parameter.

    Returns:
        SparsePCA: Trained SparsePCA model.
    """
    print("Training Sparse PCA autoencoder...")
    sparse_pca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42, max_iter=1000)
    sparse_pca.fit(data)
    print("Training completed.")
    return sparse_pca


def analyze_components(sparse_pca: SparsePCA, feature_names: list):
    """
    Analyze and visualize the learned sparse components.

    Args:
        sparse_pca (SparsePCA): Trained SparsePCA model.
        feature_names (list): List of feature names corresponding to dataset columns.
    """
    components = sparse_pca.components_
    plt.figure(figsize=(12, 8))
    plt.imshow(components, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Sparse PCA Components")
    plt.xlabel("Feature Index")
    plt.ylabel("Component Index")
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.tight_layout()
    plt.show()
    print("Components visualized.")


def map_components_to_actions(sparse_pca: SparsePCA, feature_names: list) -> dict:
    """
    Map the learned components to specific driving actions based on feature associations.

    Args:
        sparse_pca (SparsePCA): Trained SparsePCA model.
        feature_names (list): List of feature names.

    Returns:
        dict: Mapping of components to driving actions.
    """
    component_mapping = {}
    for idx, component in enumerate(sparse_pca.components_):
        # Identify features with high absolute weights in the component
        threshold = 0.5  # Threshold for significant weight
        significant_indices = np.where(np.abs(component) > threshold)[0]
        significant_features = [feature_names[i] for i in significant_indices]

        # Heuristically map features to driving actions
        actions = []
        for feature in significant_features:
            if "gas" in feature:
                actions.append("Gas Pressed")
            if "brake" in feature:
                actions.append("Brake Pressed")
            if "steeringAngle" in feature or "steeringPressed" in feature:
                actions.append("Steering Action")
            if "desiredLongControlState" in feature:
                actions.append("Longitudinal Control State")
            # Add more mappings as needed

        # Remove duplicates and store
        actions = list(set(actions))
        component_mapping[f"Component_{idx + 1}"] = actions if actions else ["Unmapped"]

    print("Mapped components to driving actions based on feature associations.")
    return component_mapping


# =====================
# Main Execution
# =====================

def main():
    """
    Main function to execute the sparse autoencoder analysis workflow.
    """
    try:
        # Step 1: Load the ONNX driving model
        model = load_onnx_model(DRIVING_MODEL_PATH)

        # Step 2: Extract layer/output names from the model
        layer_names = extract_layer_names(model)

        # Step 3: Load route logs
        log_paths = load_route_logs(ROUTE_ID)

        # Step 4: Extract features from logs
        layer_features = extract_features_from_logs(log_paths)

        # Step 5: Prepare data for the autoencoder
        dataset, feature_names = prepare_data_for_autoencoder(layer_features)

        # Step 6: Train the Sparse Autoencoder
        sparse_autoencoder = train_sparse_autoencoder(dataset, SPARSE_COMPONENTS, SPARSE_ALPHA)

        # Step 7: Analyze the learned components
        analyze_components(sparse_autoencoder, feature_names)

        # Step 8: Map components to driving actions
        component_action_mapping = map_components_to_actions(sparse_autoencoder, feature_names)
        for component, actions in component_action_mapping.items():
            print(f"{component} is associated with actions: {', '.join(actions)}")

        # Step 9: Save the mapping for future reference
        with open("component_action_mapping.pkl", "wb") as f:
            pickle.dump(component_action_mapping, f)
        print("Component-to-action mapping saved to 'component_action_mapping.pkl'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # In production, integrate with FrogPilot's logging mechanism
        # Example:
        # cloudlog.exception("Sparse Autoencoder Analysis Error")
        # sentry.capture_exception(e)
        raise


if __name__ == "__main__":
    main()