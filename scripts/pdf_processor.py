import json
import PyPDF2
import pandas as pd
import re
from typing import List, Tuple
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class PDFProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_parts = []
        for doc in documents:
            text_blocks, table_blocks = self.split_table_and_text(doc.page_content)
            text_parts.extend(text_blocks)
        documents = [Document(page_content=text) for text in text_parts]

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=3,  # 增大窗口，考虑更多上下文
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
            add_start_index=True  # 保留原文位置
        )
        chunks = text_splitter.split_documents(documents)
        text_chunks = [chunk.page_content for chunk in chunks]
        return text_chunks

    def split_table_and_text(self, text: str):
        """将文档内容分割为表格部分和文本部分"""
        lines = text.split('\n')
        table_blocks = []
        text_blocks = []
        current_block = []
        in_table = False

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if self._is_page_metadata(line_stripped):
                continue
            prev_line = lines[i - 1].strip() if i > 0 else None
            # 检测是否进入表格区域
            if self.is_table_line(line_stripped,prev_line) and not in_table:
                # 表格开始，保存之前的文本块
                if current_block:
                    text_blocks.append('\n'.join(current_block))
                current_block = [line]
                in_table = True
            elif not self.is_table_line(line_stripped,prev_line) and in_table:
                # 表格结束
                if current_block:
                    table_blocks.append('\n'.join(current_block))
                current_block = [line]
                in_table = False
            else:
                # 继续当前块
                current_block.append(line)

        # 处理最后一块
        if current_block:
            if in_table:
                table_blocks.append('\n'.join(current_block))
            else:
                text_blocks.append('\n'.join(current_block))

        return text_blocks, table_blocks

    def is_table_line(self, text: str, prev_line:str) -> bool:
        """
        判断一行是否为表格行
        """
        def has_consistent_structure(text: str, prev_line: str) -> bool:
            if not prev_line:
                return False
            # 空格分布相似性（粗略检测）
            space_pos_current = [i for i, char in enumerate(text) if char == ' ']
            space_pos_prev = [i for i, char in enumerate(prev_line) if char == ' ']

            # 检查是否有共同的空间位置（列对齐）
            common_positions = set(space_pos_current) & set(space_pos_prev)
            return (len(common_positions) >= 2)

        # 主检测逻辑
        structure_match = has_consistent_structure(text, prev_line)
        if structure_match:
            return True
        else:
            return False


    def _is_page_metadata(self, text: str) -> bool:
        """
        检测文本是否为页码或页眉等元数据
        """
        text = text.strip()

        # 模式1：纯数字页码（前后无其他内容）
        if re.match(r'^\d+$', text):
            return True

        # 模式2：股份有限公司 + 年份 + 年度报告 + 可能空格 + 页码
        company_report_patterns = [
            r'^.*股份有限公司.*\d{4}.*年度报告.*\d*$',
            r'^.*股份有限公司.*年度报告.*$',
            r'^.*\d{4}.*年度报告.*$'
        ]

        for pattern in company_report_patterns:
            if re.match(pattern, text):
                return True

        # 模式3：过短的行（可能是孤立的页码或分隔符）
        if len(text) < 5 and (text.isdigit() or text in ['-', '/', '|']):
            return True

        return False

    def visualize_chunks(self, text_chunks: List[str],filename: str):
        txt_filename = filename.replace('.pdf', '.txt')
        with open(f"../data/chunks/{txt_filename}", 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(text_chunks):
                f.write( f"Chunk #{i + 1},Length: {len(chunk)} chars\n")
                f.write(chunk + "\n")
                f.write("=" * 30 + "\n\n")
        print(f"✅ 已保存 {len(text_chunks)} 个chunks")

def main():
    processor = PDFProcessor()
    pdf_dir = "../data/raw_pdfs/"
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            chunks = processor.extract_text_from_pdf(pdf_path)
            processor.visualize_chunks(chunks, filename)

if __name__ == "__main__":
    main()

# 有两文件没保存chunks