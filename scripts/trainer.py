from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import json
import torch
import os
import random
from sklearn.model_selection import train_test_split

def prepare_data(data_dir:str):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    val_files = random.sample(json_files, int(len(json_files) * 0.1))
    train_files = [f for f in json_files if f not in val_files]
    train_triplets = []
    for file in train_files:
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            triplets = json.load(f)
            train_triplets.extend(triplets)
    val_triplets = []
    for file in val_files:
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            triplets = json.load(f)
            val_triplets.extend(triplets)
    return train_triplets, val_triplets
train_triplets, val_triplets = prepare_data("../data/json/")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_options = {
    'qwen3-0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    'bge-m3': 'bge-m3',
    'jina-embeddings-v3':'jina-embeddings-v3'
    }
model_name = model_options['qwen3-0.6B']
model = SentenceTransformer(model_name)
print(f"模型镜像: {model_name}下载完成")

class EmbeddingTrainer:
    def __init__(self, use_lora: bool = True, device_id: int = 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            print(f"使用 GPU: {self.device}")
        else:
            self.device = "cpu"
            print("使用 CPU")
        self.model = model
        self.model.to(self.device)
        self.use_lora = use_lora
        if use_lora:
            self._setup_lora()

    def _setup_lora(self):
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.15,
            bias="none",
            # task_type=TaskType.FEATURE_EXTRACTION
        )
        self.model[0].auto_model = get_peft_model(self.model[0].auto_model, lora_config)
        print("LoRA 适配器已启用")

    def prepare_training_data(self, train_triplets: list):
        train_examples = []
        for triplet in train_triplets:
            train_examples.append(InputExample(
                texts=[triplet['query'], triplet['positive'], triplet['negative']]
            ))
        return train_examples

    def train(self, train_examples, output_path: str, num_epochs: int = 3):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        steps_per_epoch = len(train_dataloader)
        warmup_steps = min(30, int(steps_per_epoch * 0.2))
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
def main():
    # 20251030
    trainer = EmbeddingTrainer()
    train_examples = trainer.prepare_training_data(train_triplets)
    trainer.train(train_examples, "../models/text_embedding")

if __name__ == "__main__":
    main()