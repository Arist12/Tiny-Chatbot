import torch
import torch.nn.functional as F
import pandas as pd
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from tqdm import tqdm

# ========== Dataset Class ==========
class MBTIClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file, tokenizer, max_length=128):
        self.df = pd.read_csv(file)
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load existing type2label
        with open("mbti2label.json", "r") as f:
            self.type2label = json.load(f)

        for _, row in self.df.iterrows():
            label = self.type2label[row['type']]
            posts = row['posts'].split("|||")
            clean_posts = [p for p in posts if not self.contains_link(p) and not self.is_only_numbers(p)]
            for post in clean_posts:
                self.samples.append((post.strip(), label))

    def contains_link(self, text):
        return 'http' in text

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
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
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

# ========== Main ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
top_k = 3  # How many top predictions to show

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
dataset = MBTIClassificationDataset("mbti_1.csv", tokenizer)
loader = DataLoader(dataset, batch_size=32)

# Load label mapping
with open("mbti2label.json", "r") as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}  # reverse mapping

# Load model
model = BERTClassifier(num_classes=len(label2idx))
model.load_state_dict(torch.load("best_bert_mbti_classifier.pt", map_location=device))
model.to(device)
model.eval()

# Inference
with torch.no_grad():
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']
        texts = batch['text']

        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)

        probs = probs.cpu()
        logits = logits.cpu()

        for i in range(probs.size(0)):
            text = texts[i]
            label_idx = labels[i].item()
            true_label = idx2label[label_idx]

            prob = probs[i]  # (num_classes,)
            topk_probs, topk_indices = torch.topk(prob, top_k)

            print(f"\n📝 Text: {text[:100]}...")  # print first 100 chars
            print(f"🎯 Ground Truth: {true_label}")

            print("🔝 Top Predictions:")
            for rank, (p, idx) in enumerate(zip(topk_probs, topk_indices), 1):
                pred_label = idx2label[idx.item()]
                print(f"    {rank}) {pred_label}: {p.item():.4f}")

