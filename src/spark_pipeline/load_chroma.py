import os
import pandas as pd
import chromadb

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PARQUET_FILE = os.path.join(DATA_PROCESSED_DIR, "embedded_chunks.parquet")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main():
    print(f"Reading compressed Parquet file from: {PARQUET_FILE}...")
    
    if not os.path.exists(PARQUET_FILE):
        print("Error: Parquet file not found. Please run ingest_spark.py first.")
        return

    df = pd.read_parquet(PARQUET_FILE)
    total_docs = len(df)
    print(f"Found {total_docs} vectorized chunks. Initializing ChromaDB client...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="langchain")

    # Prepare data arrays
    ids = df['chunk_id'].tolist()
    embeddings = df['embedding'].tolist()
    documents = df['text'].tolist()
    
    # Extract rich metadata for ChromaDB filtering capabilities
    # We convert each row's metadata columns into a dictionary
    metadatas = [
        {
            "source_file": str(row['source_file']),
            "ticker": str(row['ticker']),
            "company_name": str(row['company_name']),
            "section": str(row['section'])
        }
        for _, row in df.iterrows()
    ]

    print("Uploading data to ChromaDB via batch upsert...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Upsert completed successfully! Total documents in collection: {collection.count()}")
    print(f"Your ChromaDB is ready in: {CHROMA_PATH}")

if __name__ == "__main__":
    main()