from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file, text_split
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()
pinecone_api_key = "pcsk_3SB3Qm_M5pUAApBpMR9p1dNAuq5FHsKQBSGYXFvUSd5DhDfcj12oK8eF54XUFTQNu4kvdX"
openai_API= "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"
os.environ["PINECONE_API_KEY"] = "pcsk_3SB3Qm_M5pUAApBpMR9p1dNAuq5FHsKQBSGYXFvUSd5DhDfcj12oK8eF54XUFTQNu4kvdX"

os.environ["OPENAI_API_KEY"] = "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"
embeddings2 =download_hugging_face_embeddings()


index_name = "al-muhawir"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings2
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})


llm = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    temperature=0.4,
    max_tokens=500
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= False)

