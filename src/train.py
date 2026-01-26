#load the data:
dataset = load_dataset(
    "csv",
    data_files={
        "train": "df_train.csv",
        "test":  "df_test.csv"
    }
)

#tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    texts = [str(t) for t in batch["text"]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text", "sentiment"])
dataset.set_format("torch")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
trainer.save_model("model")
tokenizer.save_pretrained("model")

