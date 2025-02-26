import os
from huggingface_hub import HfApi
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import json
from huggingface_hub import login


#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

# Set up Hugging Face authentication
# HfApi().set_access_token(os.getenv("HF_TOKEN"))

import os
hf_token = os.getenv("HF_TOKEN")

# if not hf_token:
#     raise ValueError("Le token Hugging Face (HF_TOKEN) n'est pas défini. Ajoutez-le dans les variables d'environnement.")

login(hf_token)


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use HuggingFaceHub instead of CTransformers
llm = HuggingFaceHub(
    repo_id="meditron-7b.Q4_K_M.gguf",
    model_kwargs={
        'max_new_tokens': 512,
        'context_length': 2048,
        'repetition_penalty': 1.1,
        'temperature': 0.1,
        'top_k': 50,
        'top_p': 0.9,
        'stream': True,
        'threads': int(os.cpu_count() / 2)
    }
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = chain({"input": query})
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
    res = Response(response_data)
    return res
