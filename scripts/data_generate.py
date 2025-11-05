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
    def generate_qa_triplets(self, text_chunks: List[str], chunks_per_batch: int) -> List[Dict]:
        """生成(问, 正例, 负例)三元组"""
        all_triplets = []
        # 分批处理，避免token限制
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
        prompt = f"""
你是一个专业的数据生成助手。你的任务是**筛选出适合"填表查询"的语料**，并生成对应的问答对。

### 第一阶段：内容筛选
从以下 `{len(chunks)}` 个文档片段中，筛选出包含**具体事实信息**的内容。
**文档片段：**
{json.dumps(chunks, ensure_ascii=False, indent=2)}

**【重点关注的信息类型】**
- ✅ **基础信息**：公司名称、股票代码、成立时间、上市地点等
- ✅ **财务数据**：营收、利润、资产规模等关键指标  
- ✅ **业务描述**：主营业务、产品服务、市场定位等
- ✅ **组织信息**：管理层、分支机构、股权结构等

**【筛选标准 - 优先选择】**
- **信息完整**：包含完整句子和段落结构，有明确的业务逻辑或概念说明  
- **可读性强**：语言通顺，逻辑清晰，数字在上下文中有明确含义
- **信息量足**：包含具体的事实、数据、定义等可用于填表的信息
**【筛选标准 - 尽量避免】**
- **表格主导**：几乎全部由连续数字序列或多行数字对齐的内容构成
- **严重乱码**：大部分内容为无法识别字符或断断续续的短语片段
- **信息稀薄**：缺乏实质性内容，主要为孤立数字或格式符号
**【重要原则】**
- 如果你无法理解某部分内容，请直接忽略它
- **如果所有内容都难以理解**（比如全是表格或乱码），请跳过整个生成任务。

### 第二阶段：数据生成
基于筛选出的高质量语料，生成适合**填表查询场景**的问答对。
**【生成要求】**
- **问题要直接**：像用户在填表时需要查询特定信息
- **正例回答要精准**：直接给出表格需要填写的具体信息
- **负例回答要有迷惑性**：相同主题但信息不匹配、部分正确但不完整、相关但非所求
- **所有内容源自原文本**：不要使用额外文本
**【格式参考】**
[
  {{
    "query": "2024年中国银行的营业收入和净利润是多少？",
    "positive": "截至 2024 年末，集团资产、负债总额分别突破 35 万亿元、32 万亿元，增长 8.11%、8.20%。
全年实现营业收入和净利润6,301亿元、2,527亿元，分别增长1.16%、2.58%，集团不良贷款
率 1.25%，下降 0.02 个百分点，境外商行利润总额贡献度超过 22%，主要业务市场竞争力提
升，主要经营指标保持稳健均衡。",
    "negative": "2024年，境内商业银行业务实现营业收入4,771.28亿元，同比减少
30.62亿元，下降0.64%。"
  }},
  {{
    "query": "中国银行上市时间是？",
    "positive": "该银行于2006年率先成功在香港联交所和上海证券交易所挂牌上市，成为国内首家“A+H”上市银行。",
    "negative": "在中国金融改革进程中，中国银行完成了股份制改造并实现了公开上市。"
  }}
]
请开始执行这个两阶段任务：先筛选出包含具体信息的内容，然后生成适合填表查询的问答对。
"""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_body={"data_collection": "deny"},  # 不使用会存储数据的提供商
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
        """清理API响应，提取JSON部分"""
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
        model="google/gemini-2.5-flash-lite",
    )

    def load_all_chunks(path:str):
        with open(path, 'r', encoding='utf-8') as f:
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

    chunk_dir = "../data/chunks/"
    json_dir = "../data/json/"
    for filename in os.listdir(chunk_dir):
        chunk_path = os.path.join(chunk_dir, filename)
        chunks = load_all_chunks(chunk_path)
        test_triplets = generator.generate_qa_triplets(chunks, chunks_per_batch=16)
        if test_triplets:
            json_filename = filename.replace('.txt', '.json')
            generator.save_triplets(test_triplets,os.path.join(json_dir,json_filename))
        else:
            print("{json_filename}文件生成失败")

# 生成样本太少了，只能生成一半左右的样本，
# 很多是f"数据格式验证失败，期望{len(chunks)}个，实际{len(triplets)}个，重试 {attempt + 1}/{max_retries}")
# 还有print(f"JSON解析错误: {e}，重试 {attempt + 1}/{max_retries}")
# 5000个训练样本，500个测试集吧