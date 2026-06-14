import time
print("1. Start test_startup.py")

t0 = time.time()
import os
import sys
import logging
print(f"2. Imports standard libraries took {time.time() - t0:.4f}s")

t0 = time.time()
from fastapi import FastAPI
print(f"3. Import FastAPI took {time.time() - t0:.4f}s")

t0 = time.time()
import config
print(f"4. Import config took {time.time() - t0:.4f}s")

t0 = time.time()
import schemas
print(f"5. Import schemas took {time.time() - t0:.4f}s")

t0 = time.time()
import ai_models
print(f"6. Import ai_models took {time.time() - t0:.4f}s")

t0 = time.time()
import gemini_client
print(f"7. Import gemini_client took {time.time() - t0:.4f}s")

t0 = time.time()
import search
print(f"8. Import search took {time.time() - t0:.4f}s")

t0 = time.time()
import prompts
print(f"9. Import prompts took {time.time() - t0:.4f}s")

print("All imports completed successfully!")
