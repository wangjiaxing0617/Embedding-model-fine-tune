import json
from typing import List, Dict
import os


def create_test_data(test_data:List):
    queries = []
    corpus = []
    relevant_docs = {}
    doc_id_map = {}  # 文本 -> 索引的映射
    doc_counter = 0

    for triplet in test_data:  # 用全部数据构建corpus
        if triplet['positive'] not in doc_id_map:
            doc_id_map[triplet['positive']] = doc_counter
            corpus.append(triplet['positive'])
            doc_counter += 1
        negatives = triplet['negative']
        if isinstance(negatives, list):
            for neg in negatives:
                if neg not in doc_id_map:
                    doc_id_map[neg] = doc_counter
                    corpus.append(neg)
                    doc_counter += 1
        else:
            if negatives not in doc_id_map:
                doc_id_map[negatives] = doc_counter
                corpus.append(negatives)
                doc_counter += 1

    for i, triplet in enumerate(test_data):
        queries.append(triplet['query'])
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

def load_test_triplets(test_file_path):
    if os.path.exists(test_file_path):
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_triplets = json.load(f)
        return test_triplets
    else:
        print(f"测试集文件 {test_file_path} 不存在")
        return []

def main():
    test_triplets = load_test_triplets("../data/test/test_triplets.json")
    test_data = create_test_data(test_data=test_triplets)
    save_test_data(test_data, "../data/test/test.json")
    print(f"- 查询数量: {len(test_data['queries'])}")
    print(f"- 文档数量: {len(test_data['corpus'])}")
    print(f"- 查询-文档对: {len(test_data['relevant_docs'])}")

if __name__ == "__main__":
    main()