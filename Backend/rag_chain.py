from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import OpenAI
import os
from dotenv import load_dotenv
from model_llm.llama_interface import get_llama_response
import model_llm


# Paths
CHROMA_DIR = "docs/chroma"

def load_vectordb():
    embedding = HuggingFaceEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )
    return vectordb

def build_chain():
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(
        search_type="mmr",      # Max marginal relevance = diverse + relevant
        search_kwargs={"k": 10}
    )

    llm ="model_llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",      # simplest RAG chain
        return_source_documents=False
    )
    return chain

# Simple test function (optional)
def test_chain():
    query = "46-year-old male needs knee surgery in Pune, policy issued 3 months ago"
    chain = build_chain()
    result = chain.run(query)
    print(result)

if __name__ == "__main__":
    test_chain()
