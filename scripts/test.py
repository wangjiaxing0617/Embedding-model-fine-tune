import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
import torch
import os

class EmbeddingTester:
    def __init__(self, model_path: str, device_id: int = 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            print(f"使用 GPU: {self.device}")
        else:
            self.device = "cpu"
            print("使用 CPU")
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)
    def test_retrieval(self, test_data: dict, top_k):
        queries = test_data["queries"]
        corpus = test_data["corpus"]
        relevant_docs = test_data["relevant_docs"]
        query_embeddings = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        similarities = torch.matmul(query_embeddings, corpus_embeddings.T)
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        for query_idx, query in enumerate(queries):
            # 获取真实相关文档
            true_positives = set(relevant_docs.get(str(query_idx), []))
            # 获取预测的top_k文档
            _, top_indices = torch.topk(similarities[query_idx], k=top_k)
            predicted_positives = set(top_indices.cpu().numpy())
            # 计算指标
            y_true = [1 if i in true_positives else 0 for i in range(len(corpus))]
            y_pred = [1 if i in predicted_positives else 0 for i in range(len(corpus))]
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
        # 计算平均指标
        metrics = {
            "precision@k": np.mean(all_precisions),
            "recall@k": np.mean(all_recalls),
            "f1@k": np.mean(all_f1_scores),
            "num_queries": len(all_precisions)
        }
        return metrics

def main():
    tester = EmbeddingTester("../models/20251113")
    with open("../data/test/bishuiyuan.json", "r") as f:
        test_data = json.load(f)
    # 评估不同top_k的检索效果
    for top_k in [1, 3, 5]:
        metrics = tester.test_retrieval(test_data, top_k=top_k)
        print(f"\n--- Top{top_k} 检索指标 ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
if __name__ == "__main__":
    main()