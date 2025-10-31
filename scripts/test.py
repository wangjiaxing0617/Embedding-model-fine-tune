import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
import torch
import os

class EmbeddingEvaluator:
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

    def load_eval_data(self, eval_path: str):
        """加载评估数据"""
        with open(eval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def evaluate_retrieval(self, eval_data: dict, top_k):
        """评估检索效果"""
        queries = eval_data["queries"]
        corpus = eval_data["corpus"]
        relevant_docs = eval_data["relevant_docs"]

        # 编码所有文本
        print("编码查询和文档...")
        query_embeddings = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

        # 计算相似度
        print("计算相似度...")
        similarities = torch.matmul(query_embeddings, corpus_embeddings.T)

        all_precisions = []
        all_recalls = []
        all_f1_scores = []

        for query_idx, query in enumerate(queries):
            # 获取真实相关文档
            true_positives = set(relevant_docs.get(str(query_idx), []))

            # if not true_positives:  # 如果没有相关文档，跳过
            #     continue

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
    # 初始化评估器
    evaluator = EmbeddingEvaluator("../models/single_text_embedding")

    # 加载评估数据
    eval_data = evaluator.load_eval_data("../data/eval/test.json")

    print("开始评估检索效果...")

    # 评估不同top_k的检索效果
    for top_k in [1, 3, 5]:
        metrics = evaluator.evaluate_retrieval(eval_data, top_k=top_k)
        print(f"\n--- Top{top_k} 检索指标 ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()