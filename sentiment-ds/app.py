# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0"
)

model = joblib.load("sentiment_model_olist.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):

    if not input.text or len(input.text.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="O texto deve conter pelo menos 5 caracteres."
        )

    prediction = model.predict([input.text])[0]
    probabilities = model.predict_proba([input.text])[0]

    return {
        "previsao": prediction,
        "probabilidade": float(np.max(probabilities))
    }
