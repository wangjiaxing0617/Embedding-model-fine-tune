from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import json
import torch
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_options = {
    'qwen3-0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    'qwen3-4B': 'Qwen/Qwen3-Embedding-4B',
    }

# 选择其中一个下载和使用
model_name = model_options['qwen3-0.6B']
print(f"正在从国内镜像下载: {model_name}")
model = SentenceTransformer(model_name)
print('a')
class EmbeddingTrainer:
    def __init__(self, base_model: str = "all-MiniLM-L6-v2", use_lora: bool = False):
        print("model starts")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            base_model,
            model_kwargs={"attn_implementation": "flash_attention_2"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        print("模型成功")
        self.use_lora = use_lora
        if use_lora:
            self._setup_lora()

    def _setup_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model.get_auto_model().add_adapter(lora_config, adapter_name="lora")
        self.model.get_auto_model().set_active_adapters("lora")

    def prepare_training_data(self, data_path: str):

        with open(data_path, 'r', encoding='utf-8') as f:
            triplets = json.load(f)
        train_examples = []
        for triplet in triplets:
            train_examples.append(InputExample(
                texts=[triplet['query'], triplet['positive'], triplet['negative']]
            ))

        return train_examples

    def train(self, train_examples, output_path: str, num_epochs: int = 3):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        """训练模型"""
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # 训练配置
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )

# # 主执行脚本
# def main():
#     # 1. 处理PDF
#     processor = PDFProcessor()
#     all_chunks = []
#
#     pdf_dir = "../data/raw_pdfs/"
#     for pdf_file in os.listdir(pdf_dir):
#         if pdf_file.endswith('.pdf'):
#             chunks = processor.extract_text_from_pdf(os.path.join(pdf_dir, pdf_file))
#             all_chunks.extend(chunks)
#             print(f"处理 {pdf_file}: 得到 {len(chunks)} 个文本块")
#
#     # 保存过滤后的文本块
#     with open("../data/filtered_chunks/text_chunks.json", 'w', encoding='utf-8') as f:
#         json.dump(all_chunks, f, ensure_ascii=False, indent=2)
#
#     # 2. 生成训练数据（需要配置API key）
#     # generator = TrainingDataGenerator(api_key="your-api-key")
#     # training_data = generator.generate_qa_triplets(all_chunks[:100])  # 先试100个
#
#     # 3. 训练模型
#     trainer = EmbeddingTrainer()
#     # train_examples = trainer.prepare_training_data("../data/training_data/triplets.json")
#     # trainer.train(train_examples, "../models/finetuned_embedding")
def main():
    # 20251030
    trainer = EmbeddingTrainer()
    train_examples = trainer.prepare_training_data("../data/raw_pdfs/sample_triplets.json")
    trainer.train(train_examples, "../models/single_text_embedding")

if __name__ == "__main__":
    main()