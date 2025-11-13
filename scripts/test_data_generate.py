import json
from typing import List, Dict
from train import test_triplets

def create_test_data(test_data:List):
    # 构建评估数据格式
    queries = []
    corpus = []
    relevant_docs = {}
    # 收集所有唯一的文档
    doc_id_map = {}  # 文本 -> 索引的映射
    doc_counter = 0

    for triplet in test_data:  # 用全部数据构建corpus，确保覆盖
        if triplet['positive'] not in doc_id_map:
            doc_id_map[triplet['positive']] = doc_counter
            corpus.append(triplet['positive'])
            doc_counter += 1
        if triplet['negative'] not in doc_id_map:
            doc_id_map[triplet['negative']] = doc_counter
            corpus.append(triplet['negative'])
            doc_counter += 1
    # 为评估数据构建查询和相关文档
    for i, triplet in enumerate(test_data):
        queries.append(triplet['query'])
        # 正例文档的相关性
        positive_doc_id = doc_id_map[triplet['positive']]
        relevant_docs[str(i)] = [positive_doc_id]

    test_data = {
        "queries": queries,
        "corpus": corpus,
        "relevant_docs": relevant_docs
    }
    return test_data

def save_test_data(test_data: Dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    # 从训练数据生成测试数据
    test_data = create_test_data(test_data=test_triplets)
    # 保存测试数据
    save_test_data(test_data, "../data/test/bishuiyuan.json")
    print(f"- 查询数量: {len(test_data['queries'])}")
    print(f"- 文档数量: {len(test_data['corpus'])}")
    print(f"- 查询-文档对: {len(test_data['relevant_docs'])}")