
import pandas as pd, faiss, numpy as np, os
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Training Dataset.csv')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss.index')
TEXT_PATH  = os.path.join(os.path.dirname(__file__), 'text_chunks.txt')
MODEL_NAME ='all-MiniLM-L6-v2'

def main():
    df = pd.read_csv(DATA_PATH)
    texts = df.astype(str).apply(lambda r: ' | '.join(r.values), axis=1).tolist()
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, show_progress_bar=True).astype('float32')
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    faiss.write_index(idx, INDEX_PATH)
    with open(TEXT_PATH,'w',encoding='utf-8') as f:
        f.write('\n'.join(texts))
    print('FAISS index created.')

if __name__=='__main__':
    main()
