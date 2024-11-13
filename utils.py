import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


### Important: 1 = AI generated, 0 = human generated
def prepare_data(file_path, text_label="text_", fraud_label="label"):
    df = pd.read_csv(file_path)
    texts = df[text_label].tolist()
    # set CG to 1 and non-CG to 0
    df[fraud_label] = df[fraud_label].apply(lambda x: 1 if x == "CG" else 0)
    labels = df[fraud_label].tolist()
    cols = [col for col in df.columns if col not in [text_label, fraud_label]]
    features = df[cols].values
    return texts, labels, features, cols

def concat_files():
    first = prepare_data("fraud.csv")
    second = prepare_data("spam_50000.csv")
    texts = first[0] + second[0]
    labels = first[1] + second[1]
    # features = first[2] + second[2]
    # cols = first[3] + second[3]
    # return texts, labels, features, cols
    return texts, labels


def compute_metrics(eval_pred):
    """
    Compute metrics for the model evaluation.

    Args:
        eval_pred: tuple containing predictions and labels
            predictions: numpy array of predictions
            labels: numpy array of true labels

    Returns:
        dict: Dictionary containing accuracy and F1 scores
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }