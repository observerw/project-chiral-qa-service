import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model = AutoModelForQuestionAnswering.from_pretrained('/workspace/project-chiral-qa-service/project_chiral_qa_service/model')
tokenizer = AutoTokenizer.from_pretrained('/workspace/project-chiral-qa-service/project_chiral_qa_service/model')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer, device=device)
QA_input = {
    'question': 'What\'s my favorite food?',
    'context': 'I likes eating apple.'
}
res = nlp(QA_input)

print(res)