import os
import json
import time
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key_groq = os.environ.get("GROQ_API_KEY")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PARQUET_FILE = os.path.join(DATA_PROCESSED_DIR, "embedded_chunks.parquet")
OUTPUT_JSONL = os.path.join(DATA_PROCESSED_DIR, "agentic_distillation_dataset.jsonl")

# Textual parser
parser = StrOutputParser()

# Enhanced prompt for better mathematical and coding reasoning
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Data Engineer creating an instruction-tuning dataset to train a local Autonomous Financial AI Agent.
    Given the financial context provided (which contains tabular data or metrics), generate a complex user question that requires analytical reasoning and a mathematical calculation (e.g., finding a percentage change, a ratio, or a difference).
    
    You MUST format your entire response using ONLY the following XML-style tags. Do not use Markdown blocks (```) for the code, do not use JSON.
    
    <user_query>The complex financial question.</user_query>
    <thought>The step-by-step reasoning of how to solve the question and what variables are needed.</thought>
    <python_code>The exact python script to compute the metric. Hardcode the values extracted from the context into python variables. Always print() the final result.</python_code>"""),
    ("human", "Company: {company} ({ticker})\nFinancial context:\n{context}")
])

def extract_tag(text, tag):
    """Extract content between two XML tags."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def clean_python_code(code_str):
    """
    LLMs often ignore 'Do not use markdown' instructions.
    This function strips standard python markdown backticks if present.
    """
    code_str = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
    code_str = re.sub(r"^```\s*", "", code_str, flags=re.MULTILINE)
    return code_str.strip()

def main():
    print("Starting Agentic Distillation Dataset Generation...")
    
    if not os.path.exists(PARQUET_FILE):
        print("Error: Parquet file not found. Run the Spark pipeline first.")
        return

    llm = ChatGroq(model_name="llama-3.1-8b-instant", 
                   temperature=0.2, # Lowered slightly for more deterministic coding outputs
                   api_key=api_key_groq)
    
    chain = prompt | llm | parser

    df = pd.read_parquet(PARQUET_FILE)
    
    # 1. SMART FILTERING
    numeric_sections = [
        'KEY FINANCIAL RATIOS', 
        'INCOME STATEMENT', 
        'BALANCE SHEET'
    ]
    
    filtered_df = df[df['section'].str.contains('|'.join(numeric_sections), case=False, na=False, regex=True)]
    
    n_samples = min(100, len(filtered_df))
    
    if n_samples == 0:
        print("Not enough numeric chunks found in the dataset.")
        return
        
    df_sample = filtered_df.sample(n=n_samples, random_state=42)
    print(f"Found {len(filtered_df)} numeric chunks. Sampling {n_samples} rows for generation.")
    
    dataset = []
    
    print(f"Processing {n_samples} text chunks via Groq API...")
    for index, row in df_sample.iterrows():
        context = row['text']
        company = row['company_name']
        ticker = row['ticker']
        
        try:
            raw_result = chain.invoke({
                "context": context,
                "company": company,
                "ticker": ticker
            })
            
            # Extract fields
            user_query = extract_tag(raw_result, "user_query")
            thought = extract_tag(raw_result, "thought")
            raw_python = extract_tag(raw_result, "python_code")
            
            if not user_query or not raw_python:
                print(f"  -> Skipping chunk {row['chunk_id']}: Parsing failed. Missing XML tags.")
                continue
            
            # Clean up the python code
            python_code = clean_python_code(raw_python)
            
            # Prepare format for MLX fine-tuning (Completion format)
            system_instruction = "You are an autonomous financial agent. Think step-by-step and write Python code to perform exact calculations."
            
            formatted_text = (
                f"<|system|>\n{system_instruction}\n"
                f"<|user|>\nContext: {context}\nQuestion: {user_query}\n"
                f"<|assistant|>\n"
                f"<thought>\n{thought}\n</thought>\n"
                f"<python_code>\n{python_code}\n</python_code>\n"
            )
            
            dataset_entry = {"text": formatted_text}
            dataset.append(dataset_entry)
            print(f"  -> Generated Agentic Trajectory for chunk: {row['chunk_id']}")
            
            # Respect rate limits (Groq has high RPS, but better safe than sorry)
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"  -> Error generating output for chunk {row['chunk_id']}: {e}")

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\nAgentic dataset successfully generated! Saved {len(dataset)} valid rows to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()