
import os, faiss, numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_PATH = os.path.join(os.path.dirname(__file__),'faiss.index')
TEXT_PATH  = os.path.join(os.path.dirname(__file__),'text_chunks.txt')

_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
_texts = open(TEXT_PATH,'r',encoding='utf-8').read().splitlines() if os.path.exists(TEXT_PATH) else []

def retrieve(query:str, k:int=5):
    if _index is None:
        raise RuntimeError('Index missing. Run create_index.py')
    q_emb = _model.encode([query]).astype('float32')
    D,I = _index.search(q_emb,k)
    return [_texts[i] for i in I[0]]
