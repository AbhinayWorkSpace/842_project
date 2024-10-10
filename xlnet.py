import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
from tqdm import tqdm

class XLNetWithFeats(torch.nn.Module):
    def __init__(self, num_labels, num_feats):
        super().__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_labels)
        self.feats = torch.nn.Linear(num_feats, 64)
        self.classifier = torch.nn.Linear(self.xlnet.config.hidden_size + 64, num_labels)

    def forward(self, input_ids, attention_mask, feats):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.logits
        feat_out = self.feats(feats)
        combined = torch.cat((pooled, feat_out), dim=1)
        return self.classifier(combined)


def prepare_data(file_p):
    df = pd.read_csv(file_p)
    texts = df['text'].tolist()
    labels = df['fraud'].astype(int).tolist()

    cols = [col for col in df.columns if col not in ['text', 'fraud']]
    features = df[cols].values

    return texts, labels, features, cols


def train(model, train_loader, device, epochs=3, lr=2e-5):
    opt = AdamW(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = {'input_ids': batch[0], 'attention_mask': batch[1], 'feats': batch[2]}, batch[3]

            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = {'input_ids': batch[0], 'attention_mask': batch[1], 'feats': batch[2]}, batch[3]

            outputs = model(**inputs)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return classification_report(all_labels, all_preds)

def main():
    texts, labels, features, cols = prepare_data('fraud_engineered.csv')