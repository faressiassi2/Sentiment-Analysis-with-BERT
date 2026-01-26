from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyse de sentiment avec BERT",
    version="1.0"
)

class TextInput(BaseModel):
    text: str

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

@app.post("/predict")
def predict_sentiment(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "sentiment": label_map[result["label"]],
        "score": result["score"]
    }

