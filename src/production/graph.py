import os
import re
from typing import TypedDict, Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mlx_lm import load, generate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from src.production.tools import retrieve_financial_context, execute_python_code

load_dotenv()

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(BASE_DIR, "adapters")

# --- 1. LOCAL MLX CODER CLASS ---
class MLXQuantCoder:
    def __init__(self):
        print("[MLX] Loading local Qwen oRA model into Unified Memory...")
        self.model, self.tokenizer = load("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", adapter_path=ADAPTER_PATH)
        print("[MLX] Model loaded successfully.")
        
    def generate_code(self, question: str, context: str) -> str:
        system_instruction = "You are an autonomous financial code-generator agent. Think step-by-step and write Python code to perform exact calculations."
        prompt = (
            f"<|system|>\n{system_instruction}\n"
            f"<|user|>\nContext: {context}\nQuestion: {question}\n"
            f"<|assistant|>\n"
        )
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=512, verbose=True)

mlx_coder = MLXQuantCoder()

# --- 2. DEFINE THE GRAPH STATE ---
class AgentState(TypedDict):
    question: str
    context: str
    thought: str
    python_code: str
    execution_result: str
    final_answer: str
    router_decision: str
    evaluator_decision: str

# --- 3. STRUCTURED OUTPUT SCHEMAS FOR ROUTING ---
class RouteDecision(BaseModel):
    route: Literal["general_chat", "needs_financial_data"] = Field(
        description="Choose 'general_chat' for greetings or generic questions. Choose 'needs_financial_data' for questions about companies, revenue, or finance."
    )

class QueryEntities(BaseModel):
    tickers: List[str] = Field(
        default_factory=list, 
        description="List of stock ticker symbols mentioned in the user query (e.g., ['AAPL', 'TSLA']). If a company name is mentioned, convert it to its ticker. If no company is mentioned, return an empty list."
    )

class EvalDecision(BaseModel):
    route: Literal["insufficient_data", "can_answer_directly", "needs_math"] = Field(
        description="Choose 'insufficient_data' if the context doesn't contain the answer. Choose 'can_answer_directly' for facts like 'Who is the CEO?'. Choose 'needs_math' if percentages, sums, or comparisons are requested."
    )

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# --- 4. GRAPH NODES ---
def router_node(state: AgentState):
    print("-> [Node: Router] Classifying user intent...")
    structured_llm = llm.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_template("Analyze the user intent: {question}")
    decision = structured_llm.invoke(prompt.format(question=state["question"]))
    return {"router_decision": decision.route}

def retrieve_node(state: AgentState):
    print("-> [Node: Retrieve] Fetching context from ChromaDB...")
    
    # 1. Pre-Retrieval: Extract ticker for metadata filtering
    extractor = llm.with_structured_output(QueryEntities)
    prompt = ChatPromptTemplate.from_template(
        "Extract the company ticker symbols from the following user question.\n"
        "Question: {question}"
    )
    entities = extractor.invoke(prompt.format(question=state["question"]))
    
    extracted_tickers = [t.upper() for t in entities.tickers] if entities.tickers else None
    
    if extracted_tickers:
        print(f"[Retrieval Filter] Injecting Metadata Filter for Tickers: {extracted_tickers}")
    else:
        print("[Retrieval Filter] No specific company detected. Using pure semantic search.")

      
    # == 2. Pass the extracted ticker to our tool ===
    context = retrieve_financial_context(state["question"], tickers=extracted_tickers)
    return {"context": context}

def evaluator_node(state: AgentState):
    print("-> [Node: Evaluator] Checking if context is sufficient and if math is needed...")
    structured_llm = llm.with_structured_output(EvalDecision)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a strict data evaluator and router. Your ONLY job is to determine the route. DO NOT answer the question or do any math yourself.\n\n"
        "User Question: {question}\n"
        "Retrieved Context: {context}\n\n"
        "CRITICAL RULES:\n"
        "1. INSUFFICIENT DATA: If the requested company or the specific metric is missing from the context, output 'insufficient_data'.\n"
        "2. NEEDS MATH: If the data is present AND the user implies ANY calculation (words like 'calculate', 'growth', 'increase', 'decrease', 'difference', 'compare', 'margin', 'percentage'), you MUST output 'needs_math'. DO NOT COMPUTE IT YOURSELF.\n"
        "3. CAN ANSWER DIRECTLY: Use this ONLY if the user asks for a raw, static fact exactly as written in the text (e.g., 'What was the revenue in 2023?'). If ANY arithmetic is needed, refer to Rule 2."
    )

    decision = structured_llm.invoke(prompt.format(question=state["question"], context=state["context"]))
    print(f"   [Evaluator Decision]: {decision.route}")
    return {"evaluator_decision": decision.route}

def mlx_coder_node(state: AgentState):
    print("-> [Node: Local Coder] Generating Python code via MLX Qwen...")
    raw_output = mlx_coder.generate_code(state["question"], state["context"])
    
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_output, re.DOTALL)
    code_match = re.search(r"<python_code>(.*?)</python_code>", raw_output, re.DOTALL)
    
    thought = thought_match.group(1).strip() if thought_match else "No thought process generated."
    python_code = code_match.group(1).strip() if code_match else ""
    return {"thought": thought, "python_code": python_code}

def sandbox_node(state: AgentState):
    print("-> [Node: Sandbox] Executing generated Python code...")
    if not state["python_code"]:
        return {"execution_result": "Error: No Python code generated."}
    result = execute_python_code(state["python_code"])
    return {"execution_result": result}

def synthesizer_node(state: AgentState):
    print("-> [Node: Synthesizer] Writing final answer...")
    
    decision = state.get("evaluator_decision")
    
    if state.get("router_decision") == "general_chat":
        sys_msg = "You are a polite financial AI. Answer the user's greeting or generic non-financial question."
        
    elif decision == "insufficient_data":
        sys_msg = (
            "You are a professional financial AI assistant querying a secure local corporate database. "
            "The requested company or financial metric is NOT present in the retrieved documents.\n\n"
            "Apologize politely and state clearly that the data for this specific query is missing from the local database. \n"
            "CRITICAL RULES:\n"
            "1. DO NOT mention your 'training data', 'knowledge cutoff', or 'inability to browse the internet'.\n"
            "2. DO NOT act like a generic AI. Act like an enterprise software tool.\n"
            "3. Keep it strictly professional, concise (1-2 sentences max).\n"
            "4. Blame the absence of data entirely on the 'local database' or 'provided context'."
        )
        
    elif decision == "can_answer_directly":
        sys_msg = (
            "You are an expert financial analyst. Answer the user's question concisely using ONLY the provided context.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT copy and paste large tables or full documents.\n"
            "2. Extract ONLY the specific numbers requested (e.g., if asked for 'profits', look for 'Net Income' or 'Gross Profit').\n"
            "3. Format scientific notation into readable human terms (e.g., convert 1.12e+11 to '$112 Billion', or 1.12e+10 to '$11.2 Billion').\n"
            "4. Keep the answer short (2-3 sentences max) and professional.\n\n"
            f"Context: {state['context']}"
        )
        
    else: # needs_math
        if "Execution Error:" in state.get("execution_result", ""):
            sys_msg = (
                "You are an executive financial assistant. The internal mathematical calculation failed due to a system error. "
                "Apologize politely to the user and explain that you cannot compute the exact mathematical result at this moment. "
                "Do NOT invent the numbers."
            )
        else:
            sys_msg = (
                "You are an executive financial assistant. Your ONLY job is to report the EXACT mathematical result provided below to the user in a professional tone.\n\n"
                "CRITICAL RULES:\n"
                "1. DO NOT recalculate the numbers. Trust the 'Mathematical Result' completely.\n"
                "2. DO NOT write formulas or explain how the calculation was done.\n"
                "3. DO NOT explain that you used Python, a sandbox, or internal tools.\n"
                "4. Make the answer flow naturally, but the final number MUST be identical to the one in the Sandbox Result.\n\n"
                f"Exact Mathematical Calculation (from Sandbox): {state['execution_result']}\n"
            )
    
    prompt = ChatPromptTemplate.from_template(f"{sys_msg}\n\nUser Question: {{question}}")
    response = llm.invoke(prompt.format(question=state["question"]))
    return {"final_answer": response.content}

# --- 5. CONDITIONAL EDGES ROUTING LOGIC ---
def route_after_start(state: AgentState):
    return "synthesizer" if state["router_decision"] == "general_chat" else "retrieve"

def route_after_eval(state: AgentState):
    if state["evaluator_decision"] == "needs_math":
        return "coder"
    else:
        # If 'insufficient_data' or 'can_answer_directly', skip the code.
        return "synthesizer"

# --- 6. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("coder", mlx_coder_node)
workflow.add_node("sandbox", sandbox_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_after_start, {"synthesizer": "synthesizer", "retrieve": "retrieve"})
workflow.add_edge("retrieve", "evaluator")
workflow.add_conditional_edges("evaluator", route_after_eval, {"coder": "coder", "synthesizer": "synthesizer"})
workflow.add_edge("coder", "sandbox")
workflow.add_edge("sandbox", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()