#         prompt = f"""
# 你是一个专业的数据生成助手。你的任务分为两个阶段：**先筛选，后生成**：先筛选出你能理解的内容，然后基于筛选结果生成优质的训练数据。
#
# ### 第一阶段：内容筛选
# 你首先需要从以下 `{len(chunks)}` 个文档片段中，筛选出适合生成问答对的高质量内容。
# **文档片段：**
# {json.dumps(chunks, ensure_ascii=False, indent=2)}
# **【筛选标准 - 优先选择】**
# - **信息完整**：包含完整句子和段落结构，有明确的业务逻辑或概念说明
# - **可读性强**：语言通顺，逻辑清晰，数字在上下文中有明确含义
#
# **【筛选标准 - 尽量避免】**
# - **表格主导**：几乎全部由连续数字序列或多行数字对齐的内容构成
# - **严重乱码**：无法识别的字符或断断续续的短语片段
# - **信息稀薄**：缺乏实质性内容，主要为孤立数字或格式符号
# **【重要原则】**
# 如果你无法理解某个chunk的部分内容，请直接忽略它；
# 如果所有chunks都难以理解（比如全是表格或乱码），请跳过整个生成任务，不要生成任何内容
#
# ### 第二阶段：数据生成
# 基于你筛选出的高质量内容，生成适合训练Embedding模型的问答对数据。
# **【生成要求】**
# - **问题要自然**：像真实用户会问的
# - **正例回答要准确完整**：基于原文本逻辑清晰地回答问题
# - **负例回答要有迷惑性**：相同主题但不同答案、包含相同关键词但语义不同、部分相关但不完整
# - **所有内容源自原文本**：不要使用额外文本
# - **简明扼要**：在准确完整的条件下尽量简洁
#
# 以下是一些典型问答对示例，供你在判断时参考。
# [
#   {{
#     "query": "中国银行主营业务是？",
#     "positive": "中国银行作为大型商业银行，主营业务涵盖本外币兼营、业务品种齐全的各类金融服务。该银行于2006年率先成功在香港联交所和上海证券交易所挂牌上市，成为国内首家“A+H”上市银行。",
#     "negative": "中国银行长期作为国家外汇外贸专业银行，主要经营国家外汇管理、开展国际贸易结算、侨汇和其他非贸易外汇业务。在中国金融改革进程中，中国银行完成了股份制改造并实现了公开上市。"
#   }},
#   {{
#     "query": "中国银行的国际化情况怎么样？",
#     "positive": "中国银行是中国全球化和综合化程度最高的银行，在中国境内及境外64个国家和地区设有机构。该行拥有比较完善的全球服务网络，形成了以商业银行业务为主体，涵盖投资银行、直接投资、证券、保险、基金等多个领域的综合金融服务体系，能够为客户提供"一点接入、全球响应、综合服务"的金融解决方案。",
#     "negative": "中国银行在全面建设社会主义现代化国家的新征程上，将找准落实中央决策部署和实现自身高质量发展的结合点，当好服务双循环新发展格局的排头兵。作为拥有崇高使命感和责任感的银行，中银香港、澳门分行担任当地的发钞行，体现了其在境外的业务布局。中国银行始终恪守"为社会谋福利、为国家求富强"的历史使命，不断开创高质量发展新局面。"
#   }}
# ]
#
# """


'''
k折交叉验证：train里
import os
import json
from sklearn.model_selection import KFold
import numpy as np


def prepare_data_kfold(k_folds=5):
    data_dir = "../data/json/"

    # 获取所有json文件
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    json_files.sort()  # 确保顺序一致
    print(f"找到 {len(json_files)} 个JSON文件，进行 {k_folds} 折交叉验证")

    # 创建K折分割器
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_data = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(json_files)):
        train_files = [json_files[i] for i in train_idx]
        val_files = [json_files[i] for i in val_idx]

        # 加载训练集数据
        train_triplets = []
        for file in train_files:
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                triplets = json.load(f)
                train_triplets.extend(triplets)

        # 加载验证集数据
        val_triplets = []
        for file in val_files:
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                triplets = json.load(f)
                val_triplets.extend(triplets)

        fold_data.append({
            'fold': fold + 1,
            'train_triplets': train_triplets,
            'val_triplets': val_triplets,
            'train_files': train_files,
            'val_files': val_files
        })

        print(f"Fold {fold + 1}: 训练文件 {len(train_files)}个, 验证文件 {len(val_files)}个")
        print(f"Fold {fold + 1}: 训练三元组 {len(train_triplets)}个, 验证三元组 {len(val_triplets)}个")

    return fold_data


# 使用示例
fold_data = prepare_data_kfold(k_folds=5)

# 使用第一折数据进行训练
train_triplets = fold_data[0]['train_triplets']
val_triplets = fold_data[0]['val_triplets']
'''

