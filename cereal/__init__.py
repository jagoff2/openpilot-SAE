# cereal/__init__.py
from .serializer import Serializer, call, proxy, proxies_for, with_key
from .log import LogMessage

__all__ = ['Serializer', 'call', 'proxy', 'proxies_for', 'with_key', 'LogMessage']