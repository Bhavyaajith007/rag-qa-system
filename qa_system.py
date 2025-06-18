from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vectorstore
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# Load LLM
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = input("Ask a question: ")
result = qa_chain({"query": query})

# Show result
print("\nAnswer:\n", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata)
