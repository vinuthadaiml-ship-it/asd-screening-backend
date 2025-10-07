import requests
import pandas as pd

# Send Excel file to deployed FastAPI backend
with open("Asd-Child-Data.xlsx", "rb") as f:
    response = requests.post("https://asd-screening-api.onrender.com/predict", files={"file": f})

# Convert response to DataFrame
results = pd.DataFrame(response.json())

# Save predictions locally
results.to_excel("asd_predictions.xlsx", index=False)
print("✅ Saved predictions to asd_predictions.xlsx")

# Show sample predictions
print("\n🔍 Sample Predictions:")
print(results.head(10).to_string(index=False))

# Show high-risk cases
high_risk = results[results["Risk Level"] == "High Risk"]
print("\n🚨 High Risk Cases:")
print(high_risk.to_string(index=False))