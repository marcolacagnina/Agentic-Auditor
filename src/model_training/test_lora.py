import os
import re
from mlx_lm import load, generate

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ADAPTER_PATH = os.path.join(BASE_DIR, "adapters") 
MODEL_NAME = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"

def execute_agent_code(response_text):
    """
    Simulates the Agent runtime: extracts the code inside <python_code> 
    and executes it directly in the Python interpreter.
    """
    match = re.search(r"<python_code>(.*?)</python_code>", response_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        print("\n" + "="*50)
        print("AGENT WORKSPACE: EXECUTING GENERATED CODE...")
        print("="*50)
        try:
            exec(code)
        except Exception as e:
            print(f"Execution failed with error: {e}")
        print("="*50 + "\n")
    else:
        print("\nParsing Error: No <python_code> tags found in the response.")

def main():
    print(f"Loading base model ({MODEL_NAME}) and LoRA adapters from {ADAPTER_PATH}...")

    # Load the base model and merge the LoRA weights on the fly
    try:
        model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
    except FileNotFoundError:
        print(f"\nError: Adapters not found at {ADAPTER_PATH}.")
        return
    
    # Create a context mimicking a real table like the ones it saw during training
    test_context = """
    Company: Apple Inc. (AAPL)
    Section: ANNUAL INCOME STATEMENT
    
    2026-09-30    2025-09-30
    Total Revenue           385000000000  383285000000
    Cost Of Revenue         215000000000  214137000000
    Operating Expense       56000000000   54847000000
    """
    
    test_question = "Calculate the Gross Profit (Total Revenue minus Cost Of Revenue) for both years, and then calculate the year-over-year percentage growth of the Gross Profit."
    
    # SAME prompt format used during the training
    system_instruction = "You are an autonomous financial agent. Think step-by-step and write Python code to perform exact calculations."
    
    prompt = (
        f"<|system|>\n{system_instruction}\n"
        f"<|user|>\nContext: {test_context}\nQuestion: {test_question}\n"
        f"<|assistant|>\n"
    )
    
    print("\n--- START INFERENCE ---\n")
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=600, 
        verbose=True
    )
    
    print("\n\n--- END INFERENCE ---")
    
    execute_agent_code(response)

if __name__ == "__main__":
    main()