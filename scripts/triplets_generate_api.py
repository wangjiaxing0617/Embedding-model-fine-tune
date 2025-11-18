import requests
import json
import os
import re
import time
from typing import List, Dict
from pathlib import Path

class TripletsGenerator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_qa_triplets(self, text_chunks: List[str], chunks_per_batch: int) -> List[Dict]:
        all_triplets = []
        for i in range(0, len(text_chunks), chunks_per_batch):
            batch_chunks = text_chunks[i:i + chunks_per_batch]
            print(f"处理批次 {i // chunks_per_batch + 1}")
            triplets = self._generate_batch_triplets(batch_chunks)
            if triplets:
                all_triplets.extend(triplets)
                print(f"✓ 成功生成批内 {len(triplets)} 个样本\n")
            else:
                print(f"✗ 批次 {i // chunks_per_batch + 1} 未生成样本")
            time.sleep(2)  # 避免API限制
        return all_triplets

    def _generate_batch_triplets(self, chunks: List[str], max_retries: int = 3) -> List[Dict]:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "Triplets Generator"  # 可选
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是一个专业的数据生成专家，按照JSON格式输出。"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4000
                    },
                    # data=json.dumps({
                    #     #         "model": self.model,
                    #     #         "messages": [
                    #     #             {"role": "system", "content": "你是一个专业的数据生成专家，严格按照JSON格式输出。"},
                    #     #             {"role": "user", "content": prompt}
                    #     #         ],
                    #     #         "temperature": 0.7,
                    #     #         "provider": {
                    #     #             "data_collection": "deny"  # 不使用会存储数据的提供商
                    #     #         }
                    #     #     }),
                    timeout=60
                )

                if response.status_code != 200:
                    print(f"API请求失败: {response.status_code}, {response.text}")
                    continue

                result = response.json()
                content = result['choices'][0]['message']['content']

                # 清理响应文本，提取JSON
                cleaned_result = self._clean_json_response(content)
                triplets = json.loads(cleaned_result)

                # 验证数据格式
                if self._validate_triplets(triplets):
                    print(f"返回成功的数据: {len(triplets)}个")
                    return triplets
                else:
                    print("没有问答对生成")

            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}，重试 {attempt + 1}/{max_retries}")
            except Exception as e:
                print(f"生成数据时出错: {e}，重试 {attempt + 1}/{max_retries}")

            time.sleep(3)  # 重试前等待

        print(f"所有重试连接失败，跳过该批次")
        return []

    def _clean_json_response(self, text: str) -> str:

        # 移除可能的markdown代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # 查找JSON的开始和结束位置
        start_chars = ['[', '{']
        end_chars = [']', '}']
        start_idx = -1
        end_idx = -1

        # 找到第一个JSON开始字符
        for i, char in enumerate(text):
            if char in start_chars:
                start_idx = i
                break

        if start_idx == -1:
            return "[]"  # 没有找到JSON内容

        stack = []
        for i in range(start_idx, len(text)):
            char = text[i]
            if char in start_chars:
                stack.append(char)
            elif char in end_chars:
                if not stack:
                    break
                opening_char = stack.pop()
                # 检查括号是否匹配
                if (opening_char == '[' and char == ']') or (opening_char == '{' and char == '}'):
                    if not stack:  # 栈为空，找到最外层的结束位置
                        end_idx = i
                        break

        if start_idx != -1 and end_idx != -1:
            # 提取纯JSON部分
            pure_json = text[start_idx:end_idx + 1]
            return pure_json

        return "[]"  # 无法提取有效JSON

    def _validate_triplets(self, triplets) -> bool:
        if not isinstance(triplets, list):
            return False
        for triplet in triplets:
            if not isinstance(triplet, dict):
                return False
            if not all(key in triplet for key in ['query', 'positive', 'negative']):
                return False
            # if not isinstance(triplet['query'], str):
            #     return False
            # positive = triplet['positive']
            # if not isinstance(positive, dict):
            #     return False
            # if not all(key in positive for key in ['content', 'element_ids']):
            #     return False
            # if not isinstance(positive['content'], str):
            #     return False
            # if not isinstance(positive['element_ids'], list):
            #     return False
            # # 验证negative是字典且有正确结构
            # negative = triplet['negative']
            # if not isinstance(negative, dict):
            #     return False
            # if not all(key in negative for key in ['content', 'element_ids']):
            #     return False
            # if not isinstance(negative['content'], str):
            #     return False
            # if not isinstance(negative['element_ids'], list):
            #     return False
            if not all(isinstance(triplet[key], str) for key in ['query', 'positive', 'negative']):
                return False
        return True

    def save_triplets(self, triplets: List[Dict], filepath: str):
        #保存生成的三元组到文件
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存 {len(triplets)} 个训练样本到 {filepath}")

def main():
    generator = TripletsGenerator(
        api_key="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",
        model="google/gemini-2.5-flash-lite",
    )

    def load_all_chunks(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = []
        for chunk in data.get("chunks", []):
            chunks.append(chunk.get("content", ""))
        return chunks

    chunk_dir = "../data/company_chunks/"
    json_dir = "../data/json/"
    chunk_dir = Path(chunk_dir)
    json_dir = Path(json_dir)
    for company_path in chunk_dir.iterdir():
        company_name = company_path.name
        company_chunks = []
        for file_path in company_path.glob("*.json"):
            file_chunks = load_all_chunks(file_path)
            company_chunks.extend(file_chunks)
        triplets = generator.generate_qa_triplets(company_chunks, chunks_per_batch=16)
        if triplets:
            output_path = json_dir / f"{company_name}.json"
            generator.save_triplets(triplets, output_path)
            print(f"{company_name} 已保存 {len(triplets)} 个triplets")
        else:
            print(f"{company_name} 没有生成任何triplets")

if __name__ == "__main__":
    main()
#用0.py的prompt试试