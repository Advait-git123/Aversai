from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        response_format={"type": "json_object"}  # force JSON
    )

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
