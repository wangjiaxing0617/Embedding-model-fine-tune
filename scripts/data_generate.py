import requests
import json
import os
import re
import time
from openai import OpenAI
from typing import List, Dict

class TrainingDataGenerator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        # self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
    def generate_qa_triplets(self, text_chunks: List[str], chunks_per_batch: int = 3) -> List[Dict]:
        """生成(问, 正例, 负例)三元组"""
        all_triplets = []

        # 分批处理，避免token限制
        for i in range(0, len(text_chunks), chunks_per_batch):
            batch_chunks = text_chunks[i:i + chunks_per_batch]
            print(f"处理批次 {i // chunks_per_batch + 1}")

            triplets = self._generate_batch_triplets(batch_chunks)
            if triplets:
                all_triplets.extend(triplets)
                print(f"✓ 成功生成 {len(triplets)} 个三元组")
            else:
                print(f"✗ 批次 {i // chunks_per_batch + 1} 生成失败")

            time.sleep(2)  # 避免API限制

        return all_triplets

    def _generate_batch_triplets(self, chunks: List[str], max_retries: int = 3) -> List[Dict]:
        prompt = f"""
你是一个专业的数据生成助手。基于以下{len(chunks)}个文档片段，生成适合训练Embedding模型的数据。
文档片段：
{json.dumps(chunks, ensure_ascii=False, indent=2)}

要求：
- 严格为每个文档片段生成1个训练样本，共生成{len(chunks)}个样本
- 问题要自然，像真实用户会问的
- 正例回答要准确、完整地回答问题的文本，逻辑清晰
- 负例回答要有迷惑性：相同主题但不同答案、包含相同关键词但语义不同、部分相关但不完整
- 所有生成的语料都需要源自于原文本，不要使用额外文本
- 最终内容需要在准确完整的条件下尽量简明扼要

query生成的举例：（few-shot）
1.中国银行的股票代码是多少？
2.中国银行主营业务是？什么时候上市的？
3.中国银行的国际化情况怎么样？

返回格式：
[
  {{
    "query": "中国银行主营业务是？什么时候上市的？",
    "positive": "中国银行作为大型商业银行，主营业务涵盖本外币兼营、业务品种齐全的各类金融服务。该银行于2006年率先成功在香港联交所和上海证券交易所挂牌上市，成为国内首家“A+H”上市银行。",
    "negative": "中国银行长期作为国家外汇外贸专业银行，主要经营国家外汇管理、开展国际贸易结算、侨汇和其他非贸易外汇业务。在中国金融改革进程中，中国银行完成了股份制改造并实现了公开上市。"
  }},
  {{
    "query": "中国银行的国际化情况怎么样？",
    "positive": "中国银行是中国全球化和综合化程度最高的银行，在中国境内及境外64个国家和地区设有机构。该行拥有比较完善的全球服务网络，形成了以商业银行业务为主体，涵盖投资银行、直接投资、证券、保险、基金等多个领域的综合金融服务体系，能够为客户提供"一点接入、全球响应、综合服务"的金融解决方案。",
    "negative": "中国银行在全面建设社会主义现代化国家的新征程上，将找准落实中央决策部署和实现自身高质量发展的结合点，当好服务双循环新发展格局的排头兵。作为拥有崇高使命感和责任感的银行，中银香港、澳门分行担任当地的发钞行，体现了其在境外的业务布局。中国银行始终恪守"为社会谋福利、为国家求富强"的历史使命，不断开创高质量发展新局面。"
  }}
]

"""

        for attempt in range(max_retries):
            try:

                completion = self.client.chat.completions.create(

                    extra_body={
                        "data_collection": "deny"  # 不使用会存储数据的提供商
                    },
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=55
                )
                content = completion.choices[0].message.content
                # response = requests.post(
                #     url=self.api_url,
                #     headers={
                #         "Authorization": f"Bearer {self.api_key}",
                #         "Content-Type": "application/json"
                #     },
                #     data=json.dumps({
                #         "model": self.model,
                #         "messages": [
                #             {"role": "system", "content": "你是一个专业的数据生成专家，严格按照JSON格式输出。"},
                #             {"role": "user", "content": prompt}
                #         ],
                #         "temperature": 0.7,
                #         "provider": {
                #             "data_collection": "deny"  # 不使用会存储数据的提供商
                #         }
                #     }),
                #     timeout=60  # 60秒超时
                # )
                #
                # if response.status_code != 200:
                #     print(f"API请求失败: {response.status_code}, {response.text}")
                #     continue

                # result = response.json()
                # content = result['choices'][0]['message']['content']

                # 清理响应文本，提取JSON
                cleaned_result = self._clean_json_response(content)
                triplets = json.loads(cleaned_result)

                # 验证数据格式
                if self._validate_triplets(triplets) and len(triplets) == len(chunks):
                    return triplets
                else:
                    print(
                        f"数据格式验证失败，期望{len(chunks)}个，实际{len(triplets)}个，重试 {attempt + 1}/{max_retries}")
                    # 如果数量不匹配但格式正确，也可以考虑返回
                    if self._validate_triplets(triplets) and len(triplets) > 0:
                        print(f"返回部分成功的数据: {len(triplets)}个")
                        return triplets

            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}，重试 {attempt + 1}/{max_retries}")
            except Exception as e:
                print(f"生成数据时出错: {e}，重试 {attempt + 1}/{max_retries}")

            time.sleep(3)  # 重试前等待

        print(f"所有重试失败，跳过该批次")
        return []

    def _clean_json_response(self, text: str) -> str:
        """清理API响应，提取JSON部分"""
        # 移除可能的markdown代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # 如果以非JSON字符开头，尝试找到第一个{[
        if not text.startswith('{') and not text.startswith('['):
            match = re.search(r'[\{\[]', text)
            if match:
                text = text[match.start():]

        return text

    def _validate_triplets(self, triplets) -> bool:
        """验证生成的三元组格式"""
        if not isinstance(triplets, list):
            return False

        for triplet in triplets:
            if not isinstance(triplet, dict):
                return False
            if not all(key in triplet for key in ['query', 'positive', 'negative']):
                return False
            if not all(isinstance(triplet[key], str) for key in ['query', 'positive', 'negative']):
                return False

        return True

    def save_triplets(self, triplets: List[Dict], filepath: str):
        """保存生成的三元组到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)

        print(f"✓ 保存 {len(triplets)} 个训练样本到 {filepath}")


# 使用示例
if __name__ == "__main__":
    # 1. 从另一个文件导入text_chunks
    # 假设你的text_chunks在另一个Python文件中生成
    from pdf_processor import PDFProcessor

    # 或者直接读取之前保存的JSON文件
    # with open("../data/filtered_chunks/processed_chunks.json", 'r', encoding='utf-8') as f:
    #     text_chunks = json.load(f)

    # 2. 初始化生成器（使用OpenRouter）
    generator = TrainingDataGenerator(
        api_key="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",
        model="openai/gpt-4.1-nano",
    )


    def load_all_chunks():
        with open("../data/chunks/中国银行2024年年度报告.txt", 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = []
        current_chunk = ""
        in_chunk = False

        for line in content.split('\n'):
            if line.startswith("Chunk #"):
                if current_chunk and in_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                in_chunk = True
            elif line.startswith("=" * 30):
                if current_chunk and in_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                in_chunk = False
            elif in_chunk and line.strip():
                current_chunk += line + "\n"

        return chunks

    sample_chunks = load_all_chunks()
    test_triplets = generator.generate_qa_triplets(sample_chunks, chunks_per_batch=32)
    if test_triplets:
        generator.save_triplets(test_triplets, "../data/json/中国银行2024年年度报告.json")
        print("测试生成成功！")
        # 如果测试成功，处理全部数据
        # all_triplets = generator.generate_qa_triplets(text_chunks, chunks_per_batch=3)
        # generator.save_triplets(all_triplets, "../data/training_data/all_triplets.json")
    else:
        print("测试生成失败，请检查API配置")

# 生成样本太少了，只能生成一半左右的样本，
# 很多是f"数据格式验证失败，期望{len(chunks)}个，实际{len(triplets)}个，重试 {attempt + 1}/{max_retries}")
# 还有print(f"JSON解析错误: {e}，重试 {attempt + 1}/{max_retries}")

# 5000个训练样本，500个测试集吧