from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
import json
import os
import random
import torch
import time
import numpy as np
from datetime import timedelta

def prepare_data(data_dir: str):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    random.shuffle(json_files)
    total_files = len(json_files)
    train_count = int(total_files * 0.7)
    val_count = int(total_files * 0.15)
    train_files = json_files[:train_count]
    val_files = json_files[train_count:train_count + val_count]
    test_files = json_files[train_count + val_count:]

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

    test_triplets = []
    for file in test_files:
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            triplets = json.load(f)
            test_triplets.extend(triplets)

    print(f"数据划分: 训练集 {len(train_files)} 文件, {len(train_triplets)} 样本")
    print(f"         验证集 {len(val_files)} 文件, {len(val_triplets)} 样本")
    print(f"         测试集 {len(test_files)} 文件, {len(test_triplets)} 样本")
    return train_triplets, val_triplets, test_triplets
train_triplets, val_triplets, test_triplets = prepare_data("../data/json/")

model_options = {
    'qwen3-0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    'bge-m3': 'bge-m3',
    'jina-embeddings-v3': 'jina-embeddings-v3'
}
model_name = model_options['qwen3-0.6B']
model = SentenceTransformer(model_name)
print(f"模型镜像: {model_name}下载完成")

# 自定义数据集类
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'query': triplet['query'],
            'positive': triplet['positive'],
            'negative': triplet['negative']
        }

class EmbeddingTrainer:
    def __init__(self, use_lora: bool = True, device_id: int = 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            print(f"使用 GPU: {self.device}")
            torch.cuda.empty_cache()
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
        )
        self.model[0].auto_model = get_peft_model(self.model[0].auto_model, lora_config)
        print("LoRA 适配器已启用")

    def prepare_training_data(self, train_triplets: list):
        return TripletDataset(train_triplets)

    def prepare_validation_data(self, val_triplets: list):
        return TripletDataset(val_triplets)

    def get_embeddings_with_grad(self, texts):
        features = self.model.tokenize(texts)
        features = {key: value.to(self.device) for key, value in features.items()}
        outputs = self.model.forward(features)
        return outputs['sentence_embedding']

    def evaluate(self, val_dataset):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # 为验证集创建 DataLoader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: x  # 直接返回batch列表
        )

        with torch.no_grad():
            for batch in val_dataloader:
                # 提取文本
                anchor_texts = [item['query'] for item in batch]
                positive_texts = [item['positive'] for item in batch]
                negative_texts = [item['negative'] for item in batch]

                # 获取嵌入向量（验证时不计算梯度）
                anchor_emb = self.model.encode(anchor_texts, convert_to_tensor=True, device=self.device)
                positive_emb = self.model.encode(positive_texts, convert_to_tensor=True, device=self.device)
                negative_emb = self.model.encode(negative_texts, convert_to_tensor=True, device=self.device)

                # 计算损失
                pos_scores = torch.sum(anchor_emb * positive_emb, dim=1) * 20
                neg_scores = torch.sum(anchor_emb * negative_emb, dim=1) * 20
                scores = torch.stack([pos_scores, neg_scores], dim=1)
                labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
                loss = torch.nn.functional.cross_entropy(scores, labels)
                total_loss += loss.item()
                num_batches += 1

                # 清理验证时的缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, train_dataset, val_dataset, output_path: str, num_epochs: int = 4):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        def custom_collate_fn(batch):
            queries = [item['query'] for item in batch]
            positives = [item['positive'] for item in batch]
            negatives = [item['negative'] for item in batch]
            # 将所有文本合并成一个列表：[q1, p1, n1, q2, p2, n2, ...]
            all_texts = []
            for i in range(len(queries)):
                all_texts.extend([queries[i], positives[i], negatives[i]])
            return all_texts

        # 创建 DataLoader，使用较小的批次大小
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=2,  # 进一步减小批次大小
            shuffle=True,
            collate_fn=custom_collate_fn
        )

        # 使用梯度累积来模拟更大的批次
        gradient_accumulation_steps = 8  # 累积8个批次，相当于批次大小16
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_batch = 0
        best_val_loss = float('inf')
        last_improve = 0
        require_improvement = 2000
        start_time = time.time()
        print("开始训练...")
        for epoch in range(num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            epoch_loss = 0
            num_batches = 0
            optimizer.zero_grad()
            for i, batch_texts in enumerate(train_dataloader):
                # 获取嵌入向量（保持梯度）
                embeddings = self.get_embeddings_with_grad(batch_texts)
                # 分离查询、正例、负例的嵌入
                query_emb = embeddings[0::3]  # 第0, 3, 6...个元素
                pos_emb = embeddings[1::3]  # 第1, 4, 7...个元素
                neg_emb = embeddings[2::3]  # 第2, 5, 8...个元素
                pos_scores = torch.sum(query_emb * pos_emb, dim=1) * 20
                neg_scores = torch.sum(query_emb * neg_emb, dim=1) * 20
                scores = torch.stack([pos_scores, neg_scores], dim=1)
                labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
                original_loss = torch.nn.functional.cross_entropy(scores, labels)
                # 梯度累积
                loss = original_loss / gradient_accumulation_steps
                loss.backward()
                epoch_loss += original_loss.item()
                num_batches += 1
                total_batch += 1
                # 每 gradient_accumulation_steps 个批次更新一次参数
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                # 每40个batch输出一次训练信息（考虑到梯度累积）
                if total_batch % 40 == 0:
                    avg_train_loss = epoch_loss / num_batches
                    # 计算验证集loss
                    val_loss = self.evaluate(val_dataset)

                    # 检查是否是最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        improve = '*'
                        last_improve = total_batch
                        timestamp = int(time.time())
                        self.model.save(os.path.join(output_path, f"model_{timestamp}"))
                    else:
                        improve = ''
                    time_dif = timedelta(seconds=int(time.time() - start_time))
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.4f},  Val Loss: {2:>5.4f},  Time: {3} {4}'
                    print(msg.format(total_batch, avg_train_loss, val_loss, time_dif, improve))

                if total_batch - last_improve > require_improvement:
                    print("长时间验证集没有进步, auto-stopping...")
                    return
            # 处理最后一个不完整的梯度累积批次
            if len(train_dataset) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = epoch_loss / num_batches
            print(f'Epoch {epoch + 1} 最终平均 Loss: {avg_train_loss:.4f}')
            val_loss = self.evaluate(val_dataset)
            time_dif = timedelta(seconds=int(time.time() - start_time))
            msg = 'Epoch End: {0},  Train Loss: {1:>5.4f},  Val Loss: {2:>5.4f},  Time: {3}'
            print(msg.format(epoch + 1, avg_train_loss, val_loss, time_dif))
        print("训练完成!")

def main():
    trainer = EmbeddingTrainer()
    train_dataset = trainer.prepare_training_data(train_triplets)
    val_dataset = trainer.prepare_validation_data(val_triplets)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    trainer.train(train_dataset, val_dataset, "../models/20251113")

if __name__ == "__main__":
    main()