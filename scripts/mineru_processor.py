import json
import os
import tiktoken
from pathlib import Path
from typing import List, Dict, Any

class MineruJSONToChunks:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.element_map = {}
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    def process_table_element(self, element: Dict) -> str:
        table_md = element["md"]
        table_md = table_md.replace('""', '"-"')
        return table_md
    def process_text_element(self, element: Dict) -> str:
        text = element.get("value", "").strip()
        heading_level = element.get("headingLevel", 0)
        if heading_level > 0:
            hashes = "#" * heading_level
            text = f"{hashes} {text}"
        return text
    def create_chunks(self, elements: List[Dict]) -> List[Dict]:
        self.element_map = {element["element_id"]: element for element in elements}
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_elements = []  # 记录当前chunk包含的element_id
        for element in elements:
            if element["type"] == "table":
                content = self.process_table_element(element)
            else:
                content = self.process_text_element(element)
            if not content.strip():
                continue
            element_tokens = self.count_tokens(content)
            # 如果元素本身超过chunk_size，单独成块
            if element_tokens > self.chunk_size:
                # 先提交当前chunk（如果有内容）
                if current_chunk:
                    chunks.append({
                        "content": "\n\n".join(current_chunk),
                        "token_count": current_tokens,
                        "element_ids": current_elements.copy()
                    })
                    current_chunk = []
                    current_tokens = 0
                    current_elements = []
                # 这个超长元素单独成块
                chunks.append({
                    "content": content,
                    "token_count": element_tokens,
                    "element_ids": [element["element_id"]]
                })
                continue
            # 检查是否可以合并到当前chunk
            if current_tokens + element_tokens <= self.chunk_size:
                current_chunk.append(content)
                current_tokens += element_tokens
                current_elements.append(element["element_id"])
            else:
                # 提交当前chunk
                if current_chunk:
                    chunks.append({
                        "content": "\n\n".join(current_chunk),
                        "token_count": current_tokens,
                        "element_ids": current_elements.copy()
                    })
                # 开始新chunk
                current_chunk = [content]
                current_tokens = element_tokens
                current_elements = [element["element_id"]]
        # 处理最后一个chunk
        if current_chunk:
            chunks.append({
                "content": "\n\n".join(current_chunk),
                "token_count": current_tokens,
                "element_ids": current_elements.copy()
            })
        return chunks
    def get_element_by_id(self, element_id: int) -> Dict:
        return self.element_map.get(element_id)
    def add_overlap(self, chunks: List[Dict]) -> List[Dict]:
        if len(chunks) <= 1 or self.overlap == 0:
            return chunks
        chunks_with_overlap = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 第一个chunk不加overlap
                chunks_with_overlap.append(chunk)
                continue
            # 检查前一个chunk的最后一个element是否是table
            prev_element_ids = chunks[i - 1].get("element_ids", [])
            if prev_element_ids:
                last_element_id = prev_element_ids[-1]
                last_element = self.get_element_by_id(last_element_id)  # 需要实现这个方法
                # 如果最后一个元素是表格，则不添加overlap
                if last_element and last_element.get("type") == "table":
                    chunks_with_overlap.append(chunk)
                    continue
            prev_content = chunks[i - 1]["content"]
            prev_tokens = self.encoding.encode(prev_content)
            if len(prev_tokens) > self.overlap:
                overlap_tokens = prev_tokens[-self.overlap:]
                overlap_text = self.encoding.decode(overlap_tokens)
                new_content = overlap_text + "\n" + chunk["content"]
                new_tokens = self.overlap + chunk["token_count"]
                chunks_with_overlap.append({
                    "content": new_content,
                    "token_count": new_tokens,
                    "element_ids": chunk["element_ids"],
                    "overlap_tokens": self.overlap
                })
            else:
                chunks_with_overlap.append(chunk)
        return chunks_with_overlap
    def process_single_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            elements = sorted(data, key=lambda x: x["element_id"])
            chunks = self.create_chunks(elements)
            # 添加overlap
            chunks_with_overlap = self.add_overlap(chunks)
            output_data = {
                "source_file": input_path,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "total_chunks": len(chunks_with_overlap),
                "chunks": chunks_with_overlap
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✓ 处理完成: {input_path} (生成 {len(chunks_with_overlap)} 个chunks)")
        except Exception as e:
            print(f"✗ 处理失败: {input_path} - {str(e)}")

    def process_directory(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"输入目录不存在: {input_dir}")
            return
        # 遍历所有JSON文件
        json_files = list(input_path.rglob("*.json"))
        if not json_files:
            print(f"在 {input_dir} 中未找到JSON文件")
            return
        print(f"找到 {len(json_files)} 个JSON文件，开始处理...")
        for json_file in json_files:
            # 计算输出路径
            relative_path = json_file.relative_to(input_path)
            output_path = Path(output_dir) / relative_path
            self.process_single_file(str(json_file), str(output_path))

def main():
    INPUT_BASE_DIR = "../data/raw_data/mineru"
    OUTPUT_BASE_DIR = "../data/company_chunks"
    CHUNK_SIZE = 256
    OVERLAP = 0
    processor = MineruJSONToChunks(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )
    processor.process_directory(INPUT_BASE_DIR, OUTPUT_BASE_DIR)
if __name__ == "__main__":
    main()

# https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/markdown.py