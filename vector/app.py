import os
import uuid
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize clients and text splitter
QDRANT_URL = os.getenv("QDRANT_URL")
MODEL_NAME= os.getenv("MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

embedding_model = TextEmbedding(model_name=MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Create collection if it doesn't exist
def get_client():
    qdrant_client = QdrantClient(url=QDRANT_URL)
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=embedding_model.get_embedding_size(MODEL_NAME),  # FastEmbed default dimension
                    distance=Distance.COSINE
                )
            )
    try:    
        yield qdrant_client
    finally:
        qdrant_client.close()

class URLRequest(BaseModel):
    urls: List[str]

@app.post("/ingest")
async def ingest_url(request: URLRequest, qdrant_client=Depends(get_client)):
    try:
        # Load content from URL
        loader = WebBaseLoader(request.urls)
        documents = loader.load()
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        points = []
        
        for chunk in chunks:
            # Generate embedding
            embeddings = list(embedding_model.embed([chunk.page_content]))
            embedding_vector = embeddings[0].tolist()
            
            # Create point with payload
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding_vector,
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        return {
            "message": f"Successfully ingested {len(points)} chunked document",
            "chunks_count": len(points)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)