import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW

from combine_fraud_data import prepare_data
from fraud_utils import evaluate


class XLNetWithFeats(torch.nn.Module):
    def __init__(self, num_labels, num_feats):
        super().__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_labels)
        self.feats = torch.nn.Linear(num_feats, 64)
        self.classifier = torch.nn.Linear(self.xlnet.config.d_model + 64, num_labels)

    def forward(self, input_ids, attention_mask, feats):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        feat_out = self.feats(feats)
        combined = torch.cat((pooled, feat_out), dim=1)
        return self.classifier(combined)


class CustomDataset(Dataset):
    def __init__(self, encodings, features, labels):
        self.encodings = encodings
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['feats'] = torch.tensor(self.features[idx], dtype=torch.float)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# create another class, CustomDatasetNoFeatures, that does not include the features
class CustomDatasetNoFeatures(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def tokenize(texts, tokenizer, max_len=128):
    return tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

def train(model, train_loader, device, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            feats = batch['feats'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, feats=feats)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare data
    texts, labels, features, cols = prepare_data('fraud_engineered.csv')

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split data
    X_train, X_temp, y_train, y_temp, feats_train, feats_temp = train_test_split(texts, labels, features, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test, feats_val, feats_test = train_test_split(X_temp, y_temp, feats_temp, test_size=0.5, random_state=42)

    # Tokenize texts
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    X_train_enc = tokenize(X_train, tokenizer)
    X_val_enc = tokenize(X_val, tokenizer)
    X_test_enc = tokenize(X_test, tokenizer)

    # Create datasets
    # train_dataset = CustomDataset(X_train_enc, feats_train, y_train)
    # val_dataset = CustomDataset(X_val_enc, feats_val, y_val)
    # test_dataset = CustomDataset(X_test_enc, feats_test, y_test)

    train_dataset = CustomDatasetNoFeatures(X_train_enc, y_train)
    val_dataset = CustomDatasetNoFeatures(X_val_enc, y_val)
    test_dataset = CustomDatasetNoFeatures(X_test_enc, y_test)


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # uncomment one of these to start training
    # XLNetWithFeats needs some more work to get it working
    # model = XLNetWithFeats(num_labels=2, num_feats=features.shape[1]).to(device)
    # model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2).to(device)

    # Train model
    # this has not been run yet and might need to be checked
    # print("Training model...")
    # train(model, train_loader, device)

    # Save model
    # torch.save(model.state_dict(), 'xlnet_fraud_model.pth')
    # print("\nModel saved as 'xlnet_fraud_model.pth'")


    # uncomment this line to load the model
    # as of now, the model is saved with this path but if the whole file were to run, the path would change to xlnet_fraud_model.pth
    model = XLNetForSequenceClassification.from_pretrained('./xlnet_fraud_model', num_labels=2).to(device)


    # Evaluate model
    print("\nValidation Results:")
    val_results = evaluate(model, val_loader, device)
    print(val_results)

    print("\nTest Results:")
    test_results = evaluate(model, test_loader, device)
    print(test_results)



if __name__ == '__main__':
    main()