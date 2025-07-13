import pandas as pd, joblib, os, numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_TRAIN = os.path.join(DATA_DIR, 'train_fe.csv')
OUT_TEST = os.path.join(DATA_DIR, 'test_fe.csv')

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, 'Training Dataset.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'Test Dataset.csv'))

    y = train['Loan_Status']
    train.drop(columns=['Loan_Status'], inplace=True)

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Fill missing values
    combined.fillna({
        'Gender': 'Missing',
        'Married': 'Missing',
        'Dependents': '0',
        'Self_Employed': 'Missing',
        'Credit_History': -1,
        'LoanAmount': combined['LoanAmount'].median(),
        'Loan_Amount_Term': combined['Loan_Amount_Term'].median()
    }, inplace=True)

    # Replace '3+' with '3'
    combined['Dependents'] = combined['Dependents'].replace('3+', '3')

    # Encode categoricals
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, os.path.join(DATA_DIR, 'label_encoders.pkl'))

    # Log-transform skewed numeric features
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        combined[col] = np.log1p(combined[col])

    # Split back
    proc_train = combined.iloc[:len(train)].copy()
    proc_train['Loan_Status'] = y
    proc_test = combined.iloc[len(train):].copy()

    proc_train.to_csv(OUT_TRAIN, index=False)
    proc_test.to_csv(OUT_TEST, index=False)
    print('✅ Preprocessed datasets saved.')

if __name__ == '__main__':
    main()
