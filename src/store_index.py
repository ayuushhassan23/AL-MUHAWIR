from helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os



load_dotenv()

pinecone_API=os.environ.get('pinecone_API')

os.environ["PINECONE_API_KEY"] = "pcsk_3SB3Qm_M5pUAApBpMR9p1dNAuq5FHsKQBSGYXFvUSd5DhDfcj12oK8eF54XUFTQNu4kvdX"


extracted_data = load_pdf_file(data='DATA')
text_chunks=text_split(extracted_data)
embeddings2 =download_hugging_face_embeddings()


pinecone_api_key = "pcsk_3SB3Qm_M5pUAApBpMR9p1dNAuq5FHsKQBSGYXFvUSd5DhDfcj12oK8eF54XUFTQNu4kvdX"
pinecone_env = "us-east-1"

pc = Pinecone(api_key=pinecone_api_key)

index_name = "al-muhawir"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
    
    
    # Embed each chunk and upsert the embeddings into your Pinecone index.

    docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,   
    index_name=index_name,
    embedding=embeddings2,
    pinecone_api_key=pinecone_api_key,

)
