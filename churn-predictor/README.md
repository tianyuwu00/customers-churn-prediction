# Churn Predictor

LightGBM-based customer churn prediction service with a REST API.

## Project Structure

```
churn-predictor/
├── app/
│   ├── api.py          # FastAPI inference service
│   └── predict.py      # Preprocessing and inference logic
├── models/             # Trained model artifacts
├── tests/              # Unit tests
├── data/               # Training/test data (not tracked in git)
├── train.py            # Training pipeline
├── notebook.ipynb      # EDA and modelling notebook
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Outputs `models/lgb_model_v1.joblib`.

## Running the API

```bash
uvicorn app.api:app --reload
```

API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /health`
Liveness check.

```bash
curl http://localhost:8000/health
```

### `POST /predict`
Predict churn for a single customer.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "id": 1,
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "No",
      "DeviceProtection": "Yes",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "Yes",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 358.20
    },
    "threshold": 0.5,
    "return_proba": false
  }'
```

Response:
```json
{"id": 1, "churn_proba": 0.312, "churn": 0}
```

### `POST /predict/batch`
Predict churn for multiple customers at once.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { ...customer1... },
      { ...customer2... }
    ],
    "threshold": 0.5
  }'
```

Response:
```json
[
  {"id": 1, "churn_proba": 0.312, "churn": 0},
  {"id": 2, "churn_proba": 0.731, "churn": 1}
]
```

### Interactive Docs

Visit `http://localhost:8000/docs` for the full Swagger UI.

## Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

## Tests

```bash
pytest tests/test_predict.py -v
```
