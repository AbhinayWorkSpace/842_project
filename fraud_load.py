import pandas as pd
import numpy as np

def load_data():
    data = pd.read_csv('fraud.csv')
    data['label'] = data['label'].map({'OR': 0, 'CG': 1})
    # change label header to 'fraud'
    data = data.rename(columns={'label': 'fraud', 'text_': 'text'})
    return data

load_data()