from sklearn.metrics import classification_report
from transformers import Trainer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("model")
trainer = Trainer(model=model)

predictions = trainer.predict(trainer.eval_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(classification_report(
    y_true,
    y_pred,
    target_names=["negative", "neutral", "positive"]
))
