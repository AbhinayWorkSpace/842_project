import datetime
import os

import torch
from datasets import load_dataset, Dataset
from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

dataset = load_dataset("tomaarsen/setfit-absa-semeval-laptops")

train_cutoff = len(dataset["train"]) // 5
test_cutoff = len(dataset["train"]) // 5
test_data = dataset["train"].select(range(train_cutoff, test_cutoff+train_cutoff))

all_labels = set(dataset["train"]["label"] + dataset["test"]["label"])
all_labels = {label for label in all_labels if label}
le = LabelEncoder()
le.fit(list(all_labels))

test_data = test_data.to_list()
for data in test_data:
    data['label'] = le.transform([data['label']])[0]

test_data = Dataset.from_list(test_data)

model = AbsaModel.from_pretrained(
    "./absa-model/final_model-aspect",
    "./absa-model/final_model-polarity",
    spacy_model="en_core_web_sm",
)

args = TrainingArguments(
    num_epochs=3,
    batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = AbsaTrainer(
    model=model,
    args=args,
    eval_dataset=test_data,
)

with torch.no_grad():
    print('Beginning evaluation on test set')
    eval_results = trainer.evaluate()

    print(eval_results)
    print('Evaluation complete')

