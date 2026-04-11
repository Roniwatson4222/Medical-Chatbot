from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


#Extract data from PDF file
def load_pdf_file(data):
  loader = DirectoryLoader(data, glob ="*.pdf", loader_cls=PyPDFLoader)    
  documents =loader.load()
  return documents


# Filter the content from document 
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs : List = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs

 #Split the data into text chunks
def text_split(extracted_data):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   text_chunks =text_splitter.split_documents(extracted_data)
   return text_chunks

#Download embeddings from HuggingFace
def download_embeddings(text_chunks):
   embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
   return embeddings



