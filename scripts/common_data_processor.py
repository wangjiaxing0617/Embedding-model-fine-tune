import pandas as pd
import os
import json
import random
from pathlib import Path
def generate_triplets_from_csv_folder(folder_path, output_path):
    all_triplets = []
    folder_path = Path(folder_path)
    for file in folder_path.glob('*'):
        if file.suffix == '.csv':
            df = pd.read_csv(file)
            print(f"处理文件: {file}")
        try:
            for _, row in df.iterrows():
                question = str(row['question']).strip()
                answer_key = str(row['answer']).strip().upper()
                correct_text = str(row[answer_key]).strip()
                wrong_options = []
                for option in ['A', 'B', 'C', 'D']:
                    if option != answer_key:
                        option_text = str(row[option]).strip()
                        if option_text:
                            wrong_options.append(option_text)
                if correct_text and wrong_options:
                    # # 随机选择1-3个错误选项
                    # num_neg = random.randint(1, 3)
                    # negative_text = " ".join(random.sample(wrong_options, num_neg))
                    triplet = {
                        "query": question,
                        "positive": correct_text,
                        "negative": wrong_options
                    }
                    all_triplets.append(triplet)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            continue
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_triplets, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 成功生成 {len(all_triplets)} 个三元组，保存到 {output_path}")
    return all_triplets

def main():
    input_folder = "../data/raw_data/common_data"
    output_file = "../data/json/common.json"
    generate_triplets_from_csv_folder(input_folder, output_file)

if __name__ == "__main__":
    main()