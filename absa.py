import datetime
import os

import torch
from datasets import load_dataset, Dataset
from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder


def train_with_recovery(initial_batch_size=64):

    timestamp = datetime.datetime.now().strftime("%d_%H%M%S")
    output_dir = f"./absa-laptop-results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("tomaarsen/setfit-absa-semeval-laptops")

    train_cutoff = len(dataset["train"]) // 5
    test_cutoff = len(dataset["test"]) // 5

    train_data = dataset["train"].select(range(train_cutoff))
    val_data = dataset["train"].select(range(train_cutoff, test_cutoff + train_cutoff))
    test_data = dataset["test"].select(range(test_cutoff))

    all_labels = set(dataset["train"]["label"] + dataset["test"]["label"])
    all_labels = {label for label in all_labels if label}
    le = LabelEncoder()
    le.fit(list(all_labels))

    # Convert to normal Python lists for manipulation
    train_data = train_data.to_list()
    val_data = val_data.to_list()
    test_data = test_data.to_list()

    # Transform labels
    for data in train_data:
        data['label'] = le.transform([data['label']])[0]
    for data in val_data:
        data['label'] = le.transform([data['label']])[0]
    # for data in test_data:
    #     data['label'] = le.transform([data['label']])[0]

    # Convert back to Dataset
    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)
    test_data = Dataset.from_list(test_data)

    #train_data = train_data.select(range(15))
    #val_data = val_data.select(range(5))
    #test_data = test_data.select(range(5))

    print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    current_batch_size = initial_batch_size

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            model = AbsaModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                spacy_model='en_core_web_sm',
            )

            args = TrainingArguments(
                output_dir=output_dir,
                num_epochs=3,
                batch_size=initial_batch_size,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            final_save_path = os.path.join(output_dir, 'final_model')

            trainer = AbsaTrainer(
                model=model,
                args=args,
                train_dataset=train_data,
                eval_dataset=val_data,
            )

            trainer.train()
            print('training is done')
            # Save model to CPU before saving to disk
            model.to("cpu")
            print('model moved')
            torch.cuda.empty_cache()
            model.save_pretrained(final_save_path)
            print('model saved')

            print(trainer.evaluate(val_data))

            # Save training info
            with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
                f.write(f"Training completed at: {datetime.datetime.now()}\n")
                f.write(f"Final batch size: {current_batch_size}\n")


    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n\n\nMEMORY ERROR\n\n\n")
            model.to("cpu")
            model.save_pretrained(os.path.join(output_dir, 'error_final_model'))


if __name__ == "__main__":
    train_with_recovery()
