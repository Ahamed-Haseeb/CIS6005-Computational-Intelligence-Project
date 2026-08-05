# ProspectIQ Prediction Dashboard

Local Flask dashboard for the saved bank marketing random forest model.

## Run

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. The 2.53 GB model bundle is loaded on the first
prediction, so that request can take longer than later predictions.

## API

`POST /api/predict` accepts the 16 raw bank marketing fields. The server applies
the notebook's `balance_per_campaign` feature engineering, category encoding,
saved scaler, and model before returning the predicted class and probability.

`GET /api/health` reports whether the model file is present and loaded.
