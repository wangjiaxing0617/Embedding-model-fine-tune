import json
from pathlib import Path
#换所有query字段中的____为“什么”
def replace_underscores_in_queries(data):
    for item in data:
        if "query" in item and isinstance(item["query"], str):
            item["query"] = item["query"].replace("____", "什么")
    return data
def main():
    input_file = Path("../data/json/common.json")
    output_file = Path("../data/json/common_clean.json")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        modified_data = replace_underscores_in_queries(data)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：{input_file} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理过程中出现错误：{e}")

if __name__ == "__main__":
    main()