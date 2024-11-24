from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer
from utils import concat_files, compute_metrics

# text, labels, features, cols = concat_files()
# text, labels = concat_files([('AI_Human.csv', 'text', 'generated', 1.0)])
text, labels = concat_files()
print('Data loaded')


tokenizer = XLNetTokenizer.from_pretrained('./xlnet-fraud-model-v3')
num_labels = len(set(labels))
model = XLNetForSequenceClassification.from_pretrained('xlnet-fraud-model-v3', num_labels=num_labels)

# split data into 80:10:10 train:val:test
X_train, X_temp, y_train, y_temp = train_test_split(text, labels, test_size=0.2, random_state=57)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=57)

del X_temp, y_temp
def tokenize(data):
    return tokenizer(
        data['text'],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors=None
    )


train_data = Dataset.from_dict({
    'text': X_train,
    'labels': y_train
})

val_data = Dataset.from_dict({
    'text': X_val,
    'labels': y_val
})

test_data = Dataset.from_dict({
    'text': X_test,
    'labels': y_test
})

print('Tokenizing data')
train_data = train_data.map(tokenize, batched=True, remove_columns=['text'])
print('Train data tokenized')
val_data = val_data.map(tokenize, batched=True, remove_columns=['text'])
print('Val data tokenized')
test_data = test_data.map(tokenize, batched=True, remove_columns=['text'])
print('Test data tokenized')

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

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
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)
# print('Beginning training')
# trainer.train()
# print('Training complete')

# trainer.save_model('./xlnet-fraud-model-v3')
# tokenizer.save_pretrained('./xlnet-fraud-model-v3')
eval = trainer.evaluate(test_data)

print(eval)
