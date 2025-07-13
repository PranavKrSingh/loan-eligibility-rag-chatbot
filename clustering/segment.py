
import pandas as pd, os, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__),'..','data')
OUT_PATH = os.path.join(DATA_DIR,'cluster_labels.csv')

def main():
    df = pd.read_csv(os.path.join(DATA_DIR,'train_fe.csv'))
    X = df.drop(columns=['Loan_Status','Loan_ID'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels
    df[['Loan_ID','Cluster']].to_csv(OUT_PATH,index=False)
    print('Clusters computed and saved.')

if __name__=='__main__':
    main()
