from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,separators=["\n\n", "\n", "•", "،", ".", "؟", " "])
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



def download_hugging_face_embeddings():
    embeddings2=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  #this model return 384 dimensions
    return embeddings2

