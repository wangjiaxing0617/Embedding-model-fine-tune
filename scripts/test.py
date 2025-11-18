import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
import torch
import os
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device()}")
class EmbeddingTester:
    def __init__(self, model_path: str, device_id: int = 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            print(f"使用 GPU: {self.device}")
        else:
            self.device = "cpu"
            print("使用 CPU")
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)

    def test_retrieval(self, test_data: dict, top_ks=[1, 3, 5, 10, 15]):
        queries = test_data["queries"]
        corpus = test_data["corpus"]
        relevant_docs = test_data["relevant_docs"]

        query_embeddings = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        similarities = torch.matmul(query_embeddings, corpus_embeddings.T)

        all_metrics = {}
        for top_k in top_ks:
            hits = 0
            total_relevant = 0
            for query_idx in range(len(queries)):
                true_positives = set(relevant_docs.get(str(query_idx), []))
                _, top_indices = torch.topk(similarities[query_idx], k=top_k)
                predicted_positives = set(top_indices.cpu().numpy())
                # 命中数
                hits += len(true_positives & predicted_positives)
                total_relevant += len(true_positives)
            recall = hits / total_relevant if total_relevant > 0 else 0

            all_metrics[top_k] = {
                "recall": recall,
                "hits": hits
            }
        return all_metrics, len(queries)

def main():
    tester = EmbeddingTester("../models/20251113/model_1763454873")
    with open("../data/test/test.json", "r") as f:
        test_data = json.load(f)
    metrics, num_queries = tester.test_retrieval(test_data)
    print(f"总查询数: {num_queries}")
    for top_k in [1, 3, 5, 10, 15]:
        print(f"\n--- Top{top_k} 检索指标 ---")
        for metric, value in metrics[top_k].items():
            print(f"{metric}: {value:.4f}")

    # 测试基座模型
    print("\n=== 测试基座模型 ===")
    tester2 = EmbeddingTester("BAAI/bge-base-zh-v1.5")
    metrics2, num_queries2 = tester2.test_retrieval(test_data)
    print(f"总查询数: {num_queries2}")

    for top_k in [1, 3, 5, 10, 15]:
        print(f"\n--- Top{top_k} 检索指标 ---")
        for metric, value in metrics2[top_k].items():
            print(f"{metric}: {value:.4f}")
if __name__ == "__main__":
    main()