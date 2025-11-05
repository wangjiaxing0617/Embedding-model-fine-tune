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
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
    def filter_data(self, chunks: List[str], max_retries: int = 3) -> List[Dict]:
        prompt = f"""
你是一个专业的金融文档分析师，负责从 chunks 中筛选适合生成训练数据的内容。你的核心任务是：仔细分析并理解每个 chunk 的语义和结构，从全部 chunks 中识别并保留那些信息完整、可读性强、适合生成高质量问答对的chunks。
数据源如下：
{chunks}
【筛选标准 - 优先选择】
符合以下特征的 chunk 应优先选择：
文本主导：以大段连贯文字为主，即使包含少量表格或数字
信息完整：包含完整句子和段落结构，有明确的业务逻辑或概念说明
可读性强：语言通顺，逻辑清晰，数字在上下文中有明确含义
【筛选标准 - 尽量避免】
以下特征的 chunk 应尽量避免选择：
表格主导：几乎全部由连续数字序列或多行数字对齐的内容构成
严重乱码：大部分内容为无法识别字符或断断续续的短语片段
信息稀薄：缺乏实质性内容，主要为孤立数字或格式符号
【核心原则】
质量优先：如果你无法理解 chunk 的核心内容，建议不要保留
保留格式：请按原数据格式列出保留的 chunks
比例控制：重点关注内容的可理解性和信息密度

以下是一些典型示例，供你在判断时参考。
【优质chunk示例 - 应保留】
标签：概念清晰 | 信息完整 | 适合生成QA
示例1：
宁波银行股份有限公司 
 BANK OF NINGBO CO.,LTD. （股票代码：002142） 

示例2：
报告期内，本公司主要经营指标完成情况如下： 
（1）资产总额6,899.63 亿元，比上年末增加 819.78 亿元，增长 13.48%； 
（2）客户贷款总额 3,406.90 亿元，比上年末增加 406.00 亿元，增长 13.53%；

示例3：
公司经营范围为：吸收公众存款；发放短期、中期和长期贷款；办理国内结算；办理票据贴
现；发行金融债券；代理发行、代理兑付、承销政府债券；买卖政府债券；从事同业拆借；从事
银行卡业务；提供担保；代理收付款项及代理保险业务；提供保管箱业务；办理地方财政信用周
转使用资金的委托贷款业务；外汇存款、贷款、汇款；外币兑换；国际结算，结汇、售汇；同业
外汇拆借；外币票据的承兑和贴现；外汇担保；经中国银行业监督管理机构、中国人民银行和国
家外汇管理机关批准的其他业务。 

【表格chunk - 应排除】  
标签：表格数据 | 数字序列 | 不适合生成QA
示例1：
单位：（人民币）百万元 
项目 
2024 年 12 月31 日 2023 年 12 月31 日 
金额 占比 金额 占比 
公司贷款和垫款 822,628  55.73% 661,269  52.78% 
贷款 805,935  54.60% 648,265  51.74% 

【乱码chunk - 应排除】
标签：无法理解 | 字符异常 | 无信息价值
示例1：
按年度化计算权益的变动 
2024 年 
12 月 31 日   
2023 年 
12 月 31 日 
  (减少) / 增加   (减少) / 增加 
利率上升100 个基点 (1,895,430)   (2,301,467) 

示例2：
௜ ଳ 1954.12൙ 2022.6Ē2025.6 – – 40.00ڎ
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
                return content
            except Exception as e:
                print(f"大模型回复出错: {e}，重试 {attempt + 1}/{max_retries}")
            time.sleep(3)  # 重试前等待

        print(f"所有重试失败")
        return []

    def save_filtered_chunks(self, content, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

# 使用示例
if __name__ == "__main__":
    generator = TrainingDataGenerator(
        api_key="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",
        model="x-ai/grok-4-fast",
    )
    def load_all_chunks(chunk_path: str):
        with open(chunk_path, 'r', encoding='utf-8') as f:
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
    filtered_dir = "../data/filtered_chunks/"
    for filename in os.listdir(chunk_dir):
        chunk_path=os.path.join(chunk_dir, filename)
        all_chunks = load_all_chunks(chunk_path)
        response = generator.filter_data(all_chunks)
        if response:
            filtered_path = os.path.join(filtered_dir, filename)
            generator.save_filtered_chunks(response,filtered_path)
            print(f"{filename}过滤成功！")
        else:
            print(f"{filename}过滤失败")
