import logging
import time
import sys

# Đảm bảo in UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

print("1. Start debug_qdrant.py")
t0 = time.time()
from qdrant_client import QdrantClient
print(f"2. Import QdrantClient took {time.time() - t0:.4f}s")

print("3. Initializing QdrantClient(path='qdrant_db')...")
t1 = time.time()
try:
    client = QdrantClient(path="qdrant_db")
    print(f"4. Success! Initialization took {time.time() - t1:.4f}s")
    print("5. Collection list:", client.get_collections())
except Exception as e:
    print(f"4. Error: {e}")
