from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

def test_retrieval(query):
    embeddings = download_hugging_face_embeddings()
    index_name = "al-muhawir"

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

    docs = retriever.get_relevant_documents(query)
    print(f"Found {len(docs)} relevant docs:\n")
    for i, doc in enumerate(docs):
        print(f"--- Doc {i+1} ---")
        print(doc.page_content)
        print()

if __name__ == "__main__":
    test_query = input(" فكان كل فرق كالطود العظيم")
    test_retrieval(test_query)
