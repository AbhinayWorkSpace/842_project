from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer

from utils import concat_files, compute_metrics

# text, labels, features, cols = concat_files()
# text, labels = concat_files()

# try:
#     tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-fraud-model')
#     model = DistilBertForSequenceClassification.from_pretrained('./distilbert-fraud-model')
#     print('Model loaded from disk')
# except:
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
#     model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

def tokenize(data):
    return tokenizer(
        data['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors=None
    )


try:
    train_data = Dataset.load_from_disk('train_data')
    val_data = Dataset.load_from_disk('val_data')
    test_data = Dataset.load_from_disk('test_data')
    print('Data loaded from disk')
except:
    text, labels = concat_files([('AI_Human.csv', 'text', 'generated', 1.0)])
    print('Data loaded')
    # split data into 80:10:10 train:val:test
    X_train, X_temp, y_train, y_temp = train_test_split(text, labels, test_size=0.2, random_state=57)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=57)

    del X_temp, y_temp



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
    train_data = train_data.map(tokenize, batched=True, batch_size=32, remove_columns=['text'])
    train_data.save_to_disk('train_data')
    print('Train data tokenized')
    val_data = val_data.map(tokenize, batched=True, batch_size=32, remove_columns=['text'])
    val_data.save_to_disk('val_data')
    print('Val data tokenized')
    test_data = test_data.map(tokenize, batched=True, batch_size=32, remove_columns=['text'])
    test_data.save_to_disk('test_data')
    print('Test data tokenized')

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


training_args = TrainingArguments(
    output_dir='./distilbert-results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)
print('Beginning training')
trainer.train()
print('Training complete')

trainer.save_model('./distilbert-fraud-model')
tokenizer.save_pretrained('./distilbert-fraud-model')

print(trainer.evaluate(test_data))
