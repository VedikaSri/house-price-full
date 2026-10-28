
from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

MODEL_PATH = Path("models/house_price_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API")


class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=1)
    bathrooms: int = Field(ge=1)
    stories: int = Field(ge=1)

    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]

    parking: int = Field(ge=0)
    prefarea: Literal["yes", "no"]

    furnishingstatus: Literal[
        "furnished",
        "semi-furnished",
        "unfurnished"
    ]


@app.get("/")
def ready_to_serve():
    return {"status": "Ready to serve"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Endpoint for making house price predictions
@app.post("/predict")
def predict(data: HouseInput):

    features = pd.DataFrame([{
        "area": data.area,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "stories": data.stories,
        "mainroad": data.mainroad,
        "guestroom": data.guestroom,
        "basement": data.basement,
        "hotwaterheating": data.hotwaterheating,
        "airconditioning": data.airconditioning,
        "parking": data.parking,
        "prefarea": data.prefarea,
        "furnishingstatus": data.furnishingstatus
    }])

    prediction = float(model.predict(features)[0])

    return {
        "predicted_price": round(prediction, 2)
    }
