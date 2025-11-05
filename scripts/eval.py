import json
import random
from typing import List, Dict
from trainer import val_triplets

def create_evaluation_data(eval_data:list):
    # 构建评估数据格式
    queries = []
    corpus = []
    relevant_docs = {}
    # 收集所有唯一的文档
    doc_id_map = {}  # 文本 -> 索引的映射
    doc_counter = 0

    for triplet in eval_data:  # 用全部数据构建corpus，确保覆盖
        if triplet['positive'] not in doc_id_map:
            doc_id_map[triplet['positive']] = doc_counter
            corpus.append(triplet['positive'])
            doc_counter += 1
        if triplet['negative'] not in doc_id_map:
            doc_id_map[triplet['negative']] = doc_counter
            corpus.append(triplet['negative'])
            doc_counter += 1
    # 为评估数据构建查询和相关文档
    for i, triplet in enumerate(eval_data):
        queries.append(triplet['query'])
        # 正例文档的相关性
        positive_doc_id = doc_id_map[triplet['positive']]
        relevant_docs[str(i)] = [positive_doc_id]

    evaluation_data = {
        "queries": queries,
        "corpus": corpus,
        "relevant_docs": relevant_docs
    }

    return evaluation_data


def save_evaluation_data(eval_data: Dict, output_path: str):
    """保存评估数据"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    # 从训练数据生成评估数据
    evaluation_data = create_evaluation_data(eval_data=val_triplets)
    # 保存评估数据
    save_evaluation_data(evaluation_data, "../data/eval/eval_data.json")
    print(f"- 查询数量: {len(evaluation_data['queries'])}")
    print(f"- 文档数量: {len(evaluation_data['corpus'])}")
    print(f"- 查询-文档对: {len(evaluation_data['relevant_docs'])}")