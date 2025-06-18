from dotenv import load_dotenv
import os
import glob

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Correct package

# Load .env
load_dotenv()

# Step 1: Load documents from `data/`
text_files = glob.glob("data/*.txt")
docs = []
for file in text_files:
    loader = TextLoader(file)
    docs.extend(loader.load())

# Step 2: Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Step 3: Embed and save FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local("vectorstore")
