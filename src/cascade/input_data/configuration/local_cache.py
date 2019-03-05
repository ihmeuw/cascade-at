from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class LocalCache:
    """A small in-memory cache object. This has the same interface
    as memcached, so that we can switch it out for that, if appropriate."""
    def __init__(self, maxsize=8):
        if maxsize < 1:
            raise ValueError("maxsize should be greater than 0")
        self._store = dict()
        self._key_cnt = maxsize

    def set(self, key, value):
        if key in self._store:
            del self._store[key]
        self._store[key] = value
        if len(self._store) > self._key_cnt:
            # The Python dict is ordered, so this gets oldest.
            to_remove = next(iter(self._store))
            CODELOG.debug(f"LocalCache evicting {to_remove}.")
            del self._store[to_remove]

    def get(self, key):
        return self._store.get(key, None)
