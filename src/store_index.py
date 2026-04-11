from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineConeVectorStore

load_dotenv(override=True)



pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GEMINI_API_KEY"] = gemini_api_key


extracted_data =load_pdf_files("data")
filter_data =filter_to_minimal_docs(extracted_data)
text_chunks =text(filter_data)

embeddings = download_embeddings(text_chunks)

pc= Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
  documents=text_chunks,
  embedding=embeddings,
  index_name=index_name,
 )