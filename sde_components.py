import time
import hashlib
from collections import OrderedDict
from typing import Optional


class LRUCache:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.cache = OrderedDict()

    def _hash_key(self, key: str) -> str:
        # Generates a MD5 checksum key for uniform storage
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def get(self, raw_key: str) -> Optional[str]:
        hashed = self._hash_key(raw_key)
        if hashed not in self.cache:
            return None
        # Move accessed item to the end (Most Recently Used)
        self.cache.move_to_end(hashed)
        return self.cache[hashed]

    def put(self, raw_key: str, value: str) -> None:
        hashed = self._hash_key(raw_key)
        if hashed in self.cache:
            self.cache.move_to_end(hashed)
        self.cache[hashed] = value
        
        # If capacity exceeded, eject the Least Recently Used (first item)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class TokenBucketRateLimiter:
    def __init__(self, capacity: float = 5.0, fill_rate: float = 1.0 / 10.0):
        """
        capacity: Maximum tokens (requests) that can accumulate.
        fill_rate: Refill speed in tokens per second. (e.g. 0.1 tokens/sec = 1 token every 10 seconds).
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_update = time.time()

    def allow_request(self) -> bool:
        now = time.time()
        # Refill tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now

        # Consume 1 token if available
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
