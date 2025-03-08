#This code will ingest( push our documents to the vector store) and hre I used Pinecone db so free trial u can use it
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import AutoTokenizer, AutoModel
import torch
import pinecone


GROQ_API_KEY = "insert_api_key_here"
PINECONE_API_KEY = "insert_api_key_here"
PINECONE_INDEX = "insert_api_key_here"
NAMESPACE = "insert_api_key_here" 

# Load in word format:
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader(r"path\to\file.docx")#, encoding="utf-8") #add encode for tamil language is here
# Load the PDF file if u have
#loader = PyPDFLoader(r"C:\Users\NANDHINI\Desktop\Love_of_Bharathiyar\Coding__BHarathi\Bharathi Tamil =50pg.pdf")
print("done with textloader", loader)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


docs = loader.load_and_split(text_splitter=text_splitter)

chunks = text_splitter.split_documents(docs)
for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks to see it's working
    print(f"Chunk {i+1}: {len(chunk.page_content)} characters")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")# 768 dimensions choose this in while creating index

vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,   
    embedding=embeddings,
    index_name=PINECONE_INDEX,
    namespace=NAMESPACE
    )
vectorstore.add_documents(docs)
print("done")