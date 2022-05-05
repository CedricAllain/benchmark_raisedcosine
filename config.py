from pathlib import Path

# Define the cache directory for joblib.Memory
CACHEDIR = Path('./__cache__')
if not CACHEDIR.exists():
    CACHEDIR.mkdir(parents=True)