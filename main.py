from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# Load saved components
rf_model = joblib.load("rf_model.pkl")
imputer = joblib.load("imputer.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        # Load Excel file
        df = pd.read_excel(file.file)
        df.replace("?", pd.NA, inplace=True)

        # One-hot encode and align columns
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Impute missing values
        X = imputer.transform(df)

        # Predict autism probabilities
        probs = rf_model.predict_proba(X)[:, 1]
        results = pd.DataFrame({
            "Autism Probability (%)": (probs * 100).round(2),
            "Risk Level": ["High Risk" if prob >= 80 else "Low/Moderate Risk" for prob in (probs * 100)]
        })

        return JSONResponse(content=results.to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)