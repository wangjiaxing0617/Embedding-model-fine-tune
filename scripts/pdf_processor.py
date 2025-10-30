import json
import PyPDF2
import pandas as pd
import re
from typing import List, Tuple
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:



        loader = PyPDFLoader(pdf_path)
        documents = loader.load()


        filtered_docs = []
        for doc in documents:
            if not self._is_table(doc.page_content)and not self._is_page_metadata(doc.page_content):
                filtered_docs.append(doc)
            else:
                print(f"跳过表格内容: {doc.page_content[:100]}...")


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 递归分割顺序
        )


        chunks = text_splitter.split_documents(filtered_docs)


        text_chunks = [chunk.page_content for chunk in chunks]

        return text_chunks

    def _is_table(self, text: str) -> bool:

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if len(lines) < 2:
            return False
        if any('|' in line for line in lines[:3]):
            return True

        return False


    def _is_page_metadata(self, text: str) -> bool:

        metadata_patterns = [r'^\d+$']
        for pattern in metadata_patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def visualize_chunks(self, text_chunks: List[str], save_to_file: bool = True):
        if save_to_file:
            self._save_chunks_to_file(text_chunks)

    def _save_chunks_to_file(self, text_chunks: List[str], filename: str = "chunks_analysis.txt"):

        with open(f"../data/raw_pdfs/{filename}", 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(text_chunks):
                f.write(f"Chunk #{i + 1}    ,Length: {len(chunk)} chars,    Type: {'Table' if self._is_table(chunk) else 'Text'}\n")
                f.write(chunk + "\n")
                f.write("=" * 30 + "\n\n")
        print(f"✅ 已保存 {len(text_chunks)} 个chunks")



def main():
    processor = PDFProcessor()


    pdf_path = "../data/raw_pdfs/中国银行2024年年度报告.pdf"


    all_chunks = processor.extract_text_from_pdf(pdf_path)


    processor.visualize_chunks(all_chunks)

if __name__ == "__main__":
    main()

# 表格问题：没有识别出来表格，chunks怎么稳定在480的？不是越分越小吗