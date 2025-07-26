import os

from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient, models
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance


load_dotenv()

os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME= os.getenv("MODEL_NAME")

from langchain_google_genai import ChatGoogleGenerativeAI

embedding  = FastEmbedEmbeddings(model_name=MODEL_NAME, cache_dir="/app/models_cache")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

vectordb_client = QdrantClient(host=QDRANT_URL, port=6333)  # adjust host/port as needed

if COLLECTION_NAME not in [c.name for c in vectordb_client.get_collections().collections]:
    vectordb_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

retriever = QdrantVectorStore.from_existing_collection(
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    embedding=embedding,
    retrieval_mode=RetrievalMode.DENSE,
    distance=models.Distance.COSINE
).as_retriever()

retriever_tools = create_retriever_tool(
    retriever=retriever,
    name = "larion_document_retriever",
    description = "Retrieve larion internal information",
    
)
agent= create_react_agent(
    model=llm,
    tools=[retriever_tools],
    prompt="You are an assitant agent"
)


