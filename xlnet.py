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

def tokenize(texts, tokenizer, max_len=128):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True,
    )


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


def test(model, test_data, device):
    return evaluate(model, test_data, device)

def main():
    texts, labels, features, cols = prepare_data('fraud_engineered.csv')

    # normalize our features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # split the data into train and test data with a 90:10 split
    X_train, X_test, y_train, y_test, feats_train, feats_test = train_test_split(texts, labels, features, test_size=0.1, random_state=42)

    # split train into test data; keep 80 of the 90 percent as train and the other 10 as validation
    X_train, X_val, y_train, y_val, feats_train, feats_val = train_test_split(X_train, y_train, feats_train, test_size=0.11, random_state=42)

    # tokenizer and model
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetWithFeats(num_labels=2, num_feats=features.shape[1])

    # prepare the data
    train_enc = tokenize(X_train, tokenizer)
    val_enc = tokenize(X_val, tokenizer)

    train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(feats_train, dtype=torch.float), torch.tensor(y_train))
    val_dataset = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(feats_val, dtype=torch.float), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train(model, train_loader, device)

    print(evaluate(model, val_loader, device))


if __name__ == '__main__':
    main()
