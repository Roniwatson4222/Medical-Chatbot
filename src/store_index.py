from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
print(os.getcwd())

extracted_data = load_pdf_file("data")
filter_data =filter_to_minimal_docs(extracted_data)
text_chunks =text_split(filter_data)

embeddings = download_embeddings()

pc= Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

stats = index.describe_index_stats()
if stats.total_vector_count == 0:
  docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings, 
    index_name = index_name
)