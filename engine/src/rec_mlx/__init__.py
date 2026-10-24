import os

# MLX CUDA's LRUCache counts cumulative lifetime misses across all prefill chunk shapes
# and throws after 2 * capacity misses unless the thrashing check is disabled.
os.environ.setdefault("MLX_ENABLE_CACHE_THRASHING_CHECK", "0")
