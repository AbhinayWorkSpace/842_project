from sklearn.metrics import classification_report
import torch
from tqdm import tqdm


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # feats = batch['feats'].to(device)
            labels = batch['labels'].to(device)

            # uncomment this line to use the model with features
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, feats=feats)

            # use this line to use the model without features
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return classification_report(all_labels, all_preds)