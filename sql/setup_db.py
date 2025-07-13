
import pandas as pd, sqlite3, os

DATA_DIR = os.path.join(os.path.dirname(__file__),'..','data')
DB_PATH = os.path.join(DATA_DIR,'loans.db')

def main():
    df = pd.read_csv(os.path.join(DATA_DIR,'Training Dataset.csv'))
    con = sqlite3.connect(DB_PATH)
    df.to_sql('loans', con, if_exists='replace', index=False)
    con.commit()
    con.close()
    print('SQLite DB created at', DB_PATH)

if __name__=='__main__':
    main()
