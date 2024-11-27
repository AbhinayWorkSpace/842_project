import datetime
import os

import torch
from datasets import load_dataset, Dataset
from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

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

    inputs = ["The screen is crisp and the battery life is great but the charger broke easily",
              "The keyboard felt gross but the speakers are absolutely amazing",
              "I love that I can use Windows 7 on both my old laptop and my new laptop",
              "The laptop works well",
              "The pizza was delicious but the service wasn't great"]
    preds = model.predict(inputs)

    # For each prediction, transform the polarity number back to label
    for all_preds in preds:
        for pred in all_preds:
            pred['sentiment'] = le.inverse_transform([pred['polarity']])[0]
            del pred['polarity']  # optional - remove the numeric polarity
    print(preds)

from sklearn.metrics import f1_score


def calculate_absa_metrics(predictions, test_dataset):
    """
    Calculate F1 score for ABSA predictions.

    Args:
        predictions: List of lists containing prediction dictionaries
        test_dataset: The original test dataset with ground truth labels

    Returns:
        dict: Dictionary containing F1 score and other metrics
    """
    # Flatten predictions and get their polarities
    true_pairs = []
    pred_pairs = []

    # Get ground truth labels from test dataset
    for item in test_dataset:
        true_pairs.append((item['label'], item['span']))

    # Get predicted labels
    for pred in predictions:
        pred_pairs.append((pred['pred_polarity'], pred['span']))

    # Calculate true positives, false positives, and false negatives
    tp = len(set(true_pairs) & set(pred_pairs))  # Exact matches
    fp = len(set(pred_pairs) - set(true_pairs)) if not None else 0  # Predictions without matches
    fn = len(set(true_pairs) - set(pred_pairs)) if not None else 0  # Ground truth without matches

    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

    return metrics

print(calculate_absa_metrics(model.predict(test_data), test_data))

