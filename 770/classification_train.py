import pandas as pd
import re
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ========== Logging Setup ==========
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========== Dataset ==========
class MBTIClassificationDataset(Dataset):
    def __init__(self, file, tokenizer, max_length=128):
        self.df = pd.read_csv(file)
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        mbti_types = sorted(self.df['type'].unique())
        self.type2label = {mbti: idx for idx, mbti in enumerate(mbti_types)}

        # Save type2label mapping
        with open("mbti2label.json", "w") as f:
            json.dump(self.type2label, f)

        for _, row in self.df.iterrows():
            label = self.type2label[row['type']]
            posts = row['posts'].split("|||")
            clean_posts = [p for p in posts if not (self.contains_link(p) or self.is_only_numbers(p))]
            for post in clean_posts:
                self.samples.append((post.strip(), label))

    def contains_link(self, text):
        return re.search(r'https?://\S+', text)

    def is_only_numbers(self, text):
        return all(part.isdigit() for part in text.strip().split())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ========== Model ==========
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.classifier(cls_output)

# ========== Loss ==========
def MultiClass_loss_function(out, label):
    return F.cross_entropy(out, label.long())

# ========== Training Setup ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = MBTIClassificationDataset("mbti_1.csv", tokenizer)
num_classes = len(dataset.type2label)

train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

model = BERTClassifier(num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = MultiClass_loss_function

# ========== Training Loop ==========
best_val_acc = 0.0
epochs = 100


# for batch in tqdm(train_loader):
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['label'].to(device)
#     break
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    # avg_train_loss = loss.item()
    log_msg = f"Epoch {epoch+1} Train Loss: {avg_train_loss}"
    print(log_msg)
    logging.info(log_msg)

    # ========== Validation ==========
    model.eval()
    correct =0 
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    val_msg = f"✅ Epoch {epoch+1} Validation Accuracy: {acc}"
    print(val_msg)
    logging.info(val_msg)

    # Save best model
    if acc > best_val_acc:
        best_val_acc = acc
        torch.save(model.state_dict(), "best_bert_mbti_classifier.pt")
        logging.info("🔥 Best model saved!\n")
        print("🔥 Best model saved!\n")
