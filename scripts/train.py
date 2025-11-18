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
import gc

def prepare_data(data_dir: str):
    train_companies = ["药明康德", "格力电器", "顺丰控股", "宁德时代", "牧原股份", "科大讯飞", "招商银行"]
    val_companies = ["贵州茅台", "万达电影"]
    test_companies = ["碧水源"]

    train_triplets = []
    val_triplets = []
    test_triplets = []

    for company in train_companies:
        file_path = os.path.join(data_dir, f"{company}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                triplets = json.load(f)
                train_triplets.extend(triplets)
                print(f"训练集: {company} - {len(triplets)} 样本")

    for company in val_companies:
        file_path = os.path.join(data_dir, f"{company}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                triplets = json.load(f)
                val_triplets.extend(triplets)
                print(f"验证集: {company} - {len(triplets)} 样本")

    for company in test_companies:
        file_path = os.path.join(data_dir, f"{company}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                triplets = json.load(f)
                test_triplets.extend(triplets)
                print(f"测试集: {company} - {len(triplets)} 样本")

    common_file = os.path.join(data_dir, "common_clean.json")
    if os.path.exists(common_file):
        with open(common_file, 'r', encoding='utf-8') as f:
            common_triplets = json.load(f)

        random.shuffle(common_triplets)
        total_common = len(common_triplets)
        train_common_count = int(total_common * 0.7)
        val_common_count = int(total_common * 0.2)

        common_train = common_triplets[:train_common_count]
        common_val = common_triplets[train_common_count:train_common_count + val_common_count]
        common_test = common_triplets[train_common_count + val_common_count:]

        train_triplets.extend(common_train)
        val_triplets.extend(common_val)
        test_triplets.extend(common_test)
        print(
            f"通识数据: 训练集 {len(common_train)} 样本, 验证集 {len(common_val)} 样本, 测试集 {len(common_test)} 样本")

    print(f"\n最终数据划分:")
    print(f"训练集: {len(train_triplets)} 样本")
    print(f"验证集: {len(val_triplets)} 样本")
    print(f"测试集: {len(test_triplets)} 样本")
    test_output_path = "../data/test/test_triplets.json"
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_triplets, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存到: {test_output_path}")
    return train_triplets, val_triplets, test_triplets

train_triplets, val_triplets, test_triplets = prepare_data("../data/json/")

model_options = {
    'qwen3-0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    'bge': 'BAAI/bge-base-zh-v1.5',
    'jina': 'jinaai/jina-embeddings-v3'
}
model_name = model_options['jina']
model = SentenceTransformer(model_name)
print(f"模型镜像: {model_name}下载完成")
class EmbeddingTrainer:
    def __init__(self, use_lora: bool = True, device_id: int = 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
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
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model[0].auto_model = get_peft_model(self.model[0].auto_model, lora_config)
        print("LoRA 适配器已启用")

    def get_embeddings_with_grad(self, texts):
        self.model.train()
        with torch.cuda.amp.autocast():  # 混合精度减少内存
            features = self.model.tokenize(texts)
            features = {key: value.to(self.device) for key, value in features.items()}
            outputs = self.model.forward(features)
            embeddings = outputs['sentence_embedding']
        return embeddings

    def _cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def evaluate(self, val_dataset):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: x
        )
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    anchor_texts = [item['query'] for item in batch]
                    positive_texts = [item['positive'] for item in batch]
                    negative_texts = [item['negative'] for item in batch]

                    with torch.cuda.amp.autocast():
                        anchor_emb = self.model.encode(anchor_texts, convert_to_tensor=True, device=self.device)
                        positive_emb = self.model.encode(positive_texts, convert_to_tensor=True, device=self.device)
                        negative_emb = self.model.encode(negative_texts, convert_to_tensor=True, device=self.device)

                    pos_scores = torch.sum(anchor_emb * positive_emb, dim=1) * 20
                    neg_scores = torch.sum(anchor_emb * negative_emb, dim=1) * 20
                    scores = torch.stack([pos_scores, neg_scores], dim=1)
                    labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
                    loss = torch.nn.functional.cross_entropy(scores, labels)

                    total_loss += loss.item()
                    num_batches += 1

                    del anchor_emb, positive_emb, negative_emb, pos_scores, neg_scores, scores, labels, loss
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
                finally:
                    self._cleanup_memory()
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, train_dataset, val_dataset, output_path: str, num_epochs: int = 4):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        def custom_collate_fn(batch):
            queries = [item['query'] for item in batch]
            positives = [item['positive'] for item in batch]
            negatives = [item['negative'] for item in batch]
            all_texts = []
            for i in range(len(queries)):
                all_texts.extend([queries[i], positives[i], negatives[i]])
            return all_texts
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        gradient_accumulation_steps = 4
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_dataloader) * num_epochs // gradient_accumulation_steps
        )
        total_batch = 0
        best_val_loss = float('inf')
        last_improve = 0
        require_improvement = 2000
        start_time = time.time()
        for epoch in range(num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            epoch_loss = 0
            num_batches = 0
            self._cleanup_memory()
            optimizer.zero_grad()
            for i, batch_texts in enumerate(train_dataloader):
                embeddings = self.get_embeddings_with_grad(batch_texts)
                query_emb = embeddings[0::3]
                pos_emb = embeddings[1::3]
                neg_emb = embeddings[2::3]
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
                # 参数更新
                if (i + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self._cleanup_memory()
                if total_batch % 200 == 0:
                    avg_train_loss = epoch_loss / num_batches
                    val_loss = self.evaluate(val_dataset)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        improve = '*'
                        last_improve = total_batch
                        timestamp = int(time.time())
                        self.model.save(os.path.join(output_path, f"model_{timestamp}"))
                    else:
                        improve = ''
                    time_dif = timedelta(seconds=int(time.time() - start_time))
                    msg = 'Iter: {0:>6}, Train Loss: {1:>5.4f}, Val Loss: {2:>5.4f}, Time: {3} {4}'
                    print(msg.format(total_batch, avg_train_loss, val_loss, time_dif, improve))
                if total_batch - last_improve > require_improvement:
                    print("长时间验证集没有进步, auto-stopping...")
                    return
                del embeddings, query_emb, pos_emb, neg_emb, pos_scores, neg_scores, scores, labels, loss, original_loss
            if len(train_dataloader) % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self._cleanup_memory()
            # epoch结束统计
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            val_loss = self.evaluate(val_dataset)
            time_dif = timedelta(seconds=int(time.time() - start_time))
            print(f'Epoch {epoch + 1} 最终平均 Loss: {avg_train_loss:.4f}')
            print(f'验证集 Loss: {val_loss:.4f}, 时间: {time_dif}')
            self._cleanup_memory()
        print("训练完成!")

def main():
    trainer = EmbeddingTrainer()
    trainer.train(train_triplets, val_triplets, "../models/20251113")

if __name__ == "__main__":
    main()

#eval集可以和test格式一样，用来检测业务效果
