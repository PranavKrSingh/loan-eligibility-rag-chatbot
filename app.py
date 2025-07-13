
from retriever.retriever import retrieve
from llm.generate_response import generate_response
from models.predict import predict_single

def handle_prediction():
    print('Enter applicant details to predict (blank to skip):')
    fields = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed',
              'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
              'Credit_History','Property_Area']
    sample={}
    for f in fields:
        val = input(f'{f}: ')
        if val=='':
            return
        sample[f]=val
    print('Prediction ->', predict_single(sample))

def main():
    print('=== Loan Eligibility RAG Chatbot ===')
    print('Type "predict" for loan approval prediction.')
    print('Type "exit" to quit.\n')
    while True:
        q = input('You: ')
        if q.lower()=='exit':
            break
        if q.lower()=='predict':
            handle_prediction()
            continue
        ctx = '\n'.join(retrieve(q, k=5))
        ans = generate_response(ctx, q)
        print('Bot:', ans, '\n')

if __name__=='__main__':
    main()
