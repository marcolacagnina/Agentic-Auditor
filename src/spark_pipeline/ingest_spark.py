import os
import re
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import glob

# Disable tokenizers parallelism to avoid deadlocks with Spark workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PARQUET_OUTPUT = os.path.join(DATA_PROCESSED_DIR, "embedded_chunks.parquet")
MODEL_NAME = "all-MiniLM-L6-v2"

def parse_financial_document(filepath: str):
    """
    Custom parser to handle structured financial reports.
    Extracts metadata and uses context-aware chunking for tables.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract basic entity metadata
    company_match = re.search(r"Company Name:\s*(.+)", content)
    ticker_match = re.search(r"Ticker Symbol:\s*(.+)", content)
    
    company_name = company_match.group(1).strip() if company_match else "Unknown"
    ticker = ticker_match.group(1).strip() if ticker_match else "Unknown"

    # 2. Split document by the exact section separators
    # Matches: "=========================================\n1. SECTION NAME\n========================================="
    sections_raw = re.split(r"={40,}\n\d+\.\s*(.+?)\n={40,}", content)
    
    chunks_data = []
    filename = os.path.basename(filepath)

    # sections_raw[0] is the preamble, the rest follow [section_name, section_content, section_name, ...]
    for i in range(1, len(sections_raw), 2):
        section_name = sections_raw[i].strip()
        section_content = sections_raw[i+1].strip()

        # Strategy A: Prose text (Business Summary)
        if "SUMMARY" in section_name:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            text_chunks = splitter.split_text(section_content)
            
            for j, tc in enumerate(text_chunks):
                # Inject context directly into the text for the LLM
                enriched_text = f"Company: {company_name} ({ticker})\nSection: {section_name}\n\n{tc}"
                chunks_data.append({
                    "chunk_id": f"{ticker}_{section_name.replace(' ', '')}_chunk_{j}",
                    "text": enriched_text,
                    "source_file": filename,
                    "ticker": ticker,
                    "company_name": company_name,
                    "section": section_name
                })
                
        # Strategy B: Tabular Data (Ratios, Income Statement, Balance Sheet)
        else:
            lines = [line for line in section_content.split('\n') if line.strip()]
            if not lines:
                continue
                
            # Assume the first line of a table block contains the column headers (Years)
            header_row = lines[0]
            data_rows = lines[1:]
            
            # Group rows in batches of 10 to fit context window comfortably
            batch_size = 10
            for j in range(0, len(data_rows), batch_size):
                batch = data_rows[j:j+batch_size]
                
                # CRUCIAL: Prepend the header row to every chunk so numbers have meaning
                chunk_text = f"Company: {company_name} ({ticker})\nSection: {section_name}\n\n{header_row}\n" + "\n".join(batch)
                
                chunks_data.append({
                    "chunk_id": f"{ticker}_{section_name.replace(' ', '')}_batch_{j//batch_size}",
                    "text": chunk_text,
                    "source_file": filename,
                    "ticker": ticker,
                    "company_name": company_name,
                    "section": section_name
                })

    return chunks_data

def main():
    print("Starting Context-Aware Distributed Data Engineering Pipeline...")
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Financial_Text_Distributed_Ingestion") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .master("local[*]") \
        .getOrCreate()

    # Load and parse all txt files
    all_chunks = []
    file_pattern = os.path.join(DATA_RAW_DIR, "*.txt")
    for filepath in glob.glob(file_pattern):
        all_chunks.extend(parse_financial_document(filepath))

    if not all_chunks:
        print("No valid documents or chunks found.")
        spark.stop()
        return

    # Update Schema to include rich metadata
    schema = StructType([
        StructField("chunk_id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("source_file", StringType(), True),
        StructField("ticker", StringType(), True),
        StructField("company_name", StringType(), True),
        StructField("section", StringType(), True)
    ])

    df = spark.createDataFrame(all_chunks, schema=schema)
    print(f"DataFrame created with {df.count()} chunks. Schema:")
    df.printSchema()

    # Distributed Embedding via Pandas UDF
    @pandas_udf(ArrayType(FloatType()))
    def generate_embeddings(text_batch: pd.Series) -> pd.Series:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        embeddings = model.encode(text_batch.tolist())
        return pd.Series([vector.tolist() for vector in embeddings])

    print("Generating embeddings in parallel across Spark workers...")
    df_embedded = df.withColumn("embedding", generate_embeddings(col("text")))

    df_embedded.write.mode("overwrite").parquet(PARQUET_OUTPUT)
    print("PySpark Ingestion pipeline completed successfully!")
    spark.stop()

if __name__ == "__main__":
    main()