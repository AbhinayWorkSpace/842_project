from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer

from combine_fraud_data import concat_files

text, labels, features, cols = concat_files()

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
num_labels = len(set(labels))
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_labels)

# split data into 80:10:10 train:val:test
X_train, X_temp, y_train, y_temp = train_test_split(text, labels, test_size=0.2, random_state=57)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=57)

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )


train_data = Dataset.from_dict({
    'text': X_train,
    'label': y_train
})

val_data = Dataset.from_dict({
    'text': X_val,
    'label': y_val
})

test_data = Dataset.from_dict({
    'text': X_test,
    'label': y_test
})

train_data = train_data.map(tokenize, batched=True)
val_data = val_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./xlnet-results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

trainer.save_model('./xlnet-fraud-model-v3')
tokenizer.save_pretrained('./xlnet-fraud-model-v3')