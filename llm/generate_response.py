
import os, json
from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

def generate_response(context:str, query:str)->str:
    prompt = f"""You are a loan‑eligibility assistant. Use the context to answer the question.
    Context:
    {context}

    Question: {query}
    Answer:"""
    ans = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    return ans.split('Answer:')[-1].strip()
