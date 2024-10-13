import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, Trainer, TrainingArguments
from tqdm import tqdm

class XLNetWithFeats(torch.nn.Module):
    def __init__(self, num_labels, num_feats):
        super().__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_labels)
        self.feats = torch.nn.Linear(num_feats, 64)
        self.classifier = torch.nn.Linear(self.xlnet.config.d_model + 64, num_labels)

    def forward(self, input_ids, attention_mask, feats, labels):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        feat_out = self.feats(feats)
        combined = torch.cat((pooled, feat_out), dim=1)
        return self.classifier(combined)


# load data and return what we need from it: the review contents, the label, the features we engineered, and cols
def prepare_data(file_p):
    df = pd.read_csv(file_p)
    texts = df['text'].tolist()
    labels = df['fraud'].astype(int).tolist()

    cols = [col for col in df.columns if col not in ['text', 'fraud']]
    features = df[cols].values

    return texts, labels, features, cols

# tokenize the text data for XLNet
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
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'feats': batch[2], 'labels': batch[3]}

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            opt.step()
            opt.zero_grad()


# evulate the model's effectiveness on validation dataset
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


# just a wrapper, used to evaluate with the test data for ease of reading
def test(model, test_data, device):
    return evaluate(model, test_data, device)

# for simpler approach
class dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def main():
    texts, labels, features, cols = prepare_data('fraud_engineered.csv')
    label_map = {0: 'OR', 1: 'CG'}

    # split data into 80:10:10 train:val:test

    # with features
    # # normalize our features
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    # X_train, X_val, y_train, y_val, feats_train, feats_val = train_test_split(texts, labels, features, test_size=0.2, random_state=57)
    # X_val, X_test, y_val, y_test, feats_val, feats_test = train_test_split(X_val, y_val, feats_val, test_size=0.5, random_state=57)

    # without features
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=57)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=57)

    def tokenize_txt(txts, tkzr):
        return tkzr(txts, padding=True, truncation=True, max_length=256, return_tensors='pt', return_attention_mask=True)

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    X_train = tokenize_txt(X_train, tokenizer)
    X_val = tokenize_txt(X_val, tokenizer)
    X_test = tokenize_txt(X_test, tokenizer)

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    train_dataset = dataset(X_train, y_train)
    val_dataset = dataset(X_val, y_val)

    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

    print('Beginning training')

    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained('xlnet_fraud_model')
    # print the training accuracy
    print('Training accuracy:', test(model, train_dataset, 'cpu'))
    # save model to a file
    eval_results = trainer.evaluate()
    print(eval_results)


    # model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
    # text = ['Good quality and good price. I love the look and feel of this pillow.']
    # enco = tokenizer(text, padding='max_length', max_length=128, return_tensors='pt')
    # out = model(**enco)
    # preds = torch.argmax(out.logits, dim=1)
    # print(preds)


    # model = XLNetWithFeats(num_labels=2, num_feats=features.shape[1])
    #
    # # prepare the data
    # train_enc = tokenize(X_train, tokenizer)
    # val_enc = tokenize(X_val, tokenizer)
    #
    # train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(feats_train, dtype=torch.float), torch.tensor(y_train))
    # val_dataset = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(feats_val, dtype=torch.float), torch.tensor(y_val))
    #
    # train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
    # val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    #
    # train(model, train_loader, device)
    #
    # print(evaluate(model, val_loader, device))


if __name__ == '__main__':
    main()
