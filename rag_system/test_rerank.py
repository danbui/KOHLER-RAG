import time
print("Loading sentence_transformers...")
from sentence_transformers import CrossEncoder

print("Initializing CrossEncoder...")
t0 = time.time()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print(f"Reranker loaded in {time.time() - t0:.2f}s")

pairs = [["Bảng giá bồn cầu 2021", "Bản vẽ kỹ thuật và bảng giá bồn cầu Kohler 2021"]]
print("Running prediction...")
t1 = time.time()
scores = reranker.predict(pairs)
print(f"Prediction done in {time.time() - t1:.2f}s, scores: {scores}")
