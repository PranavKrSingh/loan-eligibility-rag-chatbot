import pandas as pd, joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'loan_model.joblib')

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'train_fe.csv'))
    X = df.drop(columns=['Loan_Status', 'Loan_ID'])
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print("✅ Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

if __name__ == '__main__':
    main()
