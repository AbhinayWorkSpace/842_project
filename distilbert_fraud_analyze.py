import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer

from utils import concat_files, compute_metrics

model = DistilBertForSequenceClassification.from_pretrained('./distilbert-fraud-model')
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-fraud-model')

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

text, labels = concat_files()

def tokenize(data):
    return tokenizer(
        data['text'],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors=None
    )


X_train, X_temp, y_train, y_temp = train_test_split(text, labels, test_size=0.2, random_state=57)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=57)

test_data = Dataset.from_dict({
    'text': X_test,
    'labels': y_test
})

test_data = test_data.map(tokenize, batched=True, remove_columns=['text'])

test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

test_results = trainer.evaluate(test_data)
print(test_results)
