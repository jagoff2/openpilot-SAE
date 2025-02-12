# cereal/serializer.py
# -*- coding: utf-8 -*-

import inspect
from functools import reduce

__all__ = ['Serializer', 'call', 'proxy', 'proxies_for', 'with_key']

class Serializer(object):
    """
    Define serializers using a class-based Pythonic DSL.
    
    Example:
    >>> class X(Serializer):
    ...     proxies_for('name', 'title', 'text')
    ...     
    ...     def sum(obj):
    ...         return obj.name + obj.title + obj.text
    ...
    >>> val = type('obj', (object,), {})()
    >>> val.name, val.title, val.text = 1, 2, 3
    >>> sorted(X(val).items())
    [('name', 1), ('sum', 6), ('text', 3), ('title', 2)]
    """
    class metaclass(type):
        def __new__(mcls, name, bases, attrs):
            for attr, value in attrs.items():
                if attr == '_postprocess' and callable(value):
                    attrs[attr] = staticmethod(value)
                if attr.startswith('_'):
                    continue
                if callable(value) and not isinstance(value, type):
                    attrs[attr] = staticmethod(value)
            return type.__new__(mcls, name, bases, attrs)
    
    @classmethod
    def _without(cls, *without_attrs):
        """
        Create a new Serializer excluding specified attributes.
        
        Example:
        >>> class X(Serializer):
        ...     proxies_for('name', 'title', 'text')
        ...
        >>> Y = X._without('name')
        >>> val = type('obj', (object,), {})()
        >>> val.name, val.title, val.text = 1, 2, 3
        >>> sorted(X(val).items())
        [('name', 1), ('text', 3), ('title', 2)]
        >>> sorted(Y(val).items())
        [('text', 3), ('title', 2)]
        """
        new_attrs = {attr: None for attr in without_attrs}
        return type(cls)(cls.__name__, (cls,), new_attrs)
    
    def _postprocess(dictionary):
        """
        Perform postprocessing on a serialized dictionary.
        By default, returns the dictionary unmodified.
        Override this method in subclasses to modify the output.
        """
        return dictionary
    
    def __new__(cls, obj):
        serialized = {}
        for attr in dir(cls):
            function = getattr(cls, attr)
            if not attr.startswith('_') and callable(function):
                key = getattr(function, 'key', attr)
                serialized[key] = function(obj)
        return cls._postprocess(serialized)


def with_key(value):
    """
    Set the `key` attribute on a function to a given value.
    
    Example:
    >>> @with_key('hello')
    ... def func(arg):
    ...     return arg
    >>> func.key
    'hello'
    """
    def decorator(function):
        function.key = value
        return function
    return decorator


def proxy(attr_path):
    """
    Return a proxy for a given attribute path.
    
    Example:
    >>> class X(Serializer):
    ...     def a(obj):
    ...         return obj.attr1.attr2
    ...
    can be shortened to:
    >>> class Y(Serializer):
    ...     a = proxy('attr1.attr2')
    """
    return lambda obj: reduce(getattr, attr_path.split('.'), obj)


def call(getter, *args, **kwargs):
    """
    Call the result of a getter with given arguments.
    
    Example:
    >>> class X(Serializer):
    ...     def a(obj):
    ...         return obj.attr.__str__()
    ...     def b(obj):
    ...         return obj.attr.method(1, 2, key=3)
    ...
    can be shortened to:
    >>> class Y(Serializer):
    ...     a = call(proxy('attr.__str__'))
    ...     b = call(proxy('attr.method'), 1, 2, key=3)
    """
    def caller_(obj):
        return getter(obj)(*args, **kwargs)
    return caller_


def proxies_for(*attrs):
    """
    Create proxies in the current namespace for several given attribute paths.
    
    Example:
    >>> class X(Serializer):
    ...     name = proxy('name')
    ...     title = proxy('title')
    ...     created_at = proxy('created_at')
    ...     text = proxy('text')
    ...
    can be shortened to:
    >>> class Y(Serializer):
    ...     proxies_for('name', 'title', 'created_at', 'text')
    """
    super_locals = inspect.currentframe(1).f_locals
    for attr in attrs:
        super_locals[attr.split('.')[-1]] = proxy(attr)