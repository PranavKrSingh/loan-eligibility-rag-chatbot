
import joblib, pandas as pd, os

DATA_DIR = os.path.join(os.path.dirname(__file__),'..','data')
MODEL_PATH = os.path.join(os.path.dirname(__file__),'..','models','loan_model.joblib')
ENCODERS_PATH = os.path.join(DATA_DIR,'label_encoders.pkl')

_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
_encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else None

def _normalise_input(d):
    """Fix casing and strip spaces so encoders recognise values."""
    d = d.copy()
    d["Gender"] = d["Gender"].strip().title()            # male -> Male
    d["Married"] = d["Married"].strip().title()          # yes -> Yes
    d["Self_Employed"] = d["Self_Employed"].strip().title()
    d["Education"] = "Graduate" if d["Education"].lower().startswith("g") else "Not Graduate"
    d["Dependents"] = d["Dependents"].replace("3+", "3").strip()  # "3+" → "3"
    d["Property_Area"] = d["Property_Area"].strip().title()
    return d

def predict_single(sample: dict):
    """sample: dict of raw inputs same as columns"""
    if _model is None or _encoders is None:
        return 'Model not ready. Train first.'

    # Step 1: Clean up input
    sample = _normalise_input(sample)

    # Step 2: Convert to DataFrame
    df = pd.DataFrame([sample])

    # Step 3: Apply label encoders
    for col, le in _encoders.items():
        try:
            df[col] = le.transform(df[col])
        except ValueError:
            return f'Invalid value for column "{col}": {df[col].values[0]}'

    # Step 4: Predict
    X = df.drop(columns=['Loan_ID']) if 'Loan_ID' in df.columns else df
    pred = _model.predict(X)[0]
    return 'Approved' if pred == 1 else 'Rejected'

