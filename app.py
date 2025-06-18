import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load environment variables
load_dotenv()

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vectorstore
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# Set up HuggingFace model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.title("RAG-based Q&A System")

query = st.text_input("Enter your question:")

if query:
    result = qa_chain({"query": query})
    st.write("### Answer:")
    st.write(result["result"])

    st.write("### Source Documents:")
    for i, doc in enumerate(result["source_documents"], start=1):
        st.markdown(f"**Source {i}:**")
        st.text(doc.page_content[:300] + "...")
