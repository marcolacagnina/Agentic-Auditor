import os
from io import StringIO
from contextlib import redirect_stdout
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings
)

def retrieve_financial_context(query: str, tickers: list = None) -> str:
    """Search in the DB the top k chunks, applying strict metadata filtering if tickers are provided."""
    try:
        search_kwargs = {"k": 3}
        
        if tickers and len(tickers) > 0:
            if len(tickers) == 1:
                search_kwargs["filter"] = {"ticker": tickers[0]}
            else:
                search_kwargs["filter"] = {"ticker": {"$in": tickers}}
                
        docs = vector_store.similarity_search(query, **search_kwargs)
        
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
        if not retrieved_context.strip():
            return "No relevant context found in the database for this query."
            
        return retrieved_context
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

def execute_python_code(code: str) -> str:
    """Execute code in a secure sandbox and catch the output."""
    local_variables = {}
    stdout_buffer = StringIO()
    
    # Restrict builtins to prevent malicious OS operations (No import, open, eval, etc.)
    safe_builtins = {
        "print": print,
        "abs": abs, "min": min, "max": max, "sum": sum, "round": round
    }
    
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, {"__builtins__": safe_builtins}, local_variables)
            
        execution_output = stdout_buffer.getvalue().strip()
        if not execution_output:
            return "Execution successful, but no output printed."
        return execution_output
    except Exception as e:
        return f"Execution Error: {str(e)}"