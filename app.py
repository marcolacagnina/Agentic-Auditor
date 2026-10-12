import os
import uuid
import re
from dotenv import load_dotenv

load_dotenv(override=True) 

api_key = os.getenv("LANGCHAIN_API_KEY", "")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = api_key
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Agentic_Auditor")

import streamlit as st
import time
from src.production.graph import app as agent_app

st.set_page_config(page_title="AI Auditor", layout="wide")

# Generate unique ID for the utent session such that the LangGraph's memory works
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_local_1"
    
config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("Agentic Auditor")
st.markdown("""
This dashboard shows a **Hybrid Edge-Cloud** architecture.

- Financial documents are retrieved from a local vector database (Chroma).
- Intent classification, retrieval routing, context evaluation, and final answer synthesis use Groq's cloud API.
- Python code generation runs locally using Qwen2.5-Coder-3B-Instruct via MLX.
- Generated code is executed in a restricted execution environment.
""")

with st.sidebar:
    st.header("🧠 Agent Architecture")
    st.markdown("Live DAG (Directed Acyclic Graph) state.")
    
    try:
        raw_mermaid = agent_app.get_graph().draw_mermaid()
        
        # Clean HTML graph inserted by LangGraph in the nodes.
        clean_mermaid = re.sub(r'<[^>]+>', '', raw_mermaid)
        
        custom_styles = """
        %% Styling globale
        classDef default fill:transparent,stroke:#888,stroke-width:2px;
        classDef cloudNode fill:#E1F5FE,stroke:#03A9F4,stroke-width:3px,color:#000;
        classDef localNode fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px,color:#000;
        classDef dbNode fill:#FFF3E0,stroke:#FF9800,stroke-width:3px,color:#000;
        classDef toolNode fill:#FFEBEE,stroke:#F44336,stroke-width:3px,color:#000;

        class router,evaluator,synthesizer cloudNode;
        class coder localNode;
        class retrieve dbNode;
        class sandbox toolNode;
        """
        
        # Use cleaner version that the raw_mermaid version.
        styled_mermaid = clean_mermaid + "\n" + custom_styles
        st.markdown(f"```mermaid\n{styled_mermaid}\n```")
        
        st.markdown("""
        **Legend:**
- 🔵 Cloud Reasoning (Groq)
- 🟠 Vector DB (Chroma)
- 🟣 Local Code Gen (MLX)
- 🔴 Execution Sandbox
        """)
    except Exception as e:
        st.warning(f"Error drawing graph: {e}")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Chat")
    user_query = st.text_area("Ask a question:", 
                              value="Calculate the percentage increase in Apple's Total Revenue from the fiscal year ending 2024-09-30 to the fiscal year ending 2025-09-30.")
    submit = st.button("Run", type="primary")

# Control current state of the graph and understand if we expect the human approvation.
current_state = agent_app.get_state(config)
is_waiting = current_state.next and current_state.next[0] == "sandbox"

# ---------------------------------------------------------
# The user press Run (Execution before the pause)
# ---------------------------------------------------------
if submit and not is_waiting:
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with col2:
        st.subheader("LangGraph Execution Trace")
        status_box = st.info("Initializing Agent State...")
        state = {"question": user_query}
        
        try:
            for step in agent_app.stream(state, config=config):
                if "router" in step:
                    decision = step['router']['router_decision']
                    if decision == "general_chat":
                        status_box.info("[Router] General chat detected. Bypassing database...")
                    else:
                        status_box.info("[Router] Financial query detected. Routing to Database...")
                        
                elif "retrieve" in step:
                    status_box.success("Context retrieved from ChromaDB.")
                    
                elif "evaluator" in step:
                    eval_dec = step['evaluator']['evaluator_decision']
                    if eval_dec == "insufficient_data":
                        status_box.warning("[Evaluator] Data not found. Initiating graceful fallback...")
                    elif eval_dec == "can_answer_directly":
                        status_box.success("[Evaluator] Math not needed. Synthesizing directly from text...")
                    else:
                        status_box.info("[Evaluator] Complex math required. Waking up MLX Coder...")
                        
                elif "coder" in step:
                    status_box.warning("⚠️ [Node: MLX Coder] Code generated. PAUSED for Human Review.")
                    with st.expander("Show Internal Thought Process", expanded=True):
                        st.markdown(f"**Thought:**\n{step['coder']['thought']}")
                        st.code(step['coder']['python_code'], language='python')
                        
                # Se non passa per coder (es. general chat), andrà diretto al synthesizer
                elif "synthesizer" in step:
                    status_box.success("Pipeline completed!")
                    with col1:
                        st.markdown("---")
                        st.subheader("Final Answer")
                        st.write(step["synthesizer"]["final_answer"])
                        
            # Check if the graph is in pause phase AFTER the loop.
            current_state = agent_app.get_state(config)
            is_paused = current_state.next and current_state.next[0] == "sandbox"

            if is_paused:
                st.rerun()
            
        except Exception as e:
            st.error(f"Pipeline Error: {str(e)}")


# ---------------------------------------------------------
# If graph is in pause phase, show actions bottoms
# ---------------------------------------------------------
if is_waiting:
    with col2:
        st.subheader("Human-in-the-Loop Review")
        
        # Create a visual container for UI approvation
        review_container = st.empty()
        
        # Put all advices and bottoms within this visual container
        with review_container.container():
            
            # Recover thought
            generated_thought = current_state.values.get("thought", "N/A")
            generated_code = current_state.values.get("python_code", "N/A")
            
            # Re-insert expander to show how local model reasoned
            status_box = st.info("[Node: MLX Coder] Local model generated code. Waiting for review.")
            with st.expander("Show Internal Thought Process", expanded=True):
                st.markdown(f"**Thought:**\n{generated_thought}")
            
            st.warning("⚠️ The agent wants to execute the following Python code within the Sandbox. Approve?")
            
            # Show the code to approve
            st.code(generated_code, language="python")
            
            col_btn1, col_btn2 = st.columns(2)
            approve = col_btn1.button("✅ Approve & Execute", type="primary")
            reject = col_btn2.button("❌ Reject")
        
    if approve:
        review_container.empty() 
        with col2:
            status_box = st.info("Resuming execution...")
            with st.spinner("Executing Python Sandbox..."):
                for step in agent_app.stream(None, config=config):
                    if "sandbox" in step:
                        status_box.error("[Node: Sandbox] Executing generated Python code...")
                        with st.expander("Show Sandbox Output", expanded=True):
                            st.text(step['sandbox']['execution_result'])
                    elif "synthesizer" in step:
                        status_box.success("Pipeline completed!")
                        with col1:
                            st.markdown("---")
                            st.subheader("Final Answer")
                            st.write(step["synthesizer"]["final_answer"])

    elif reject:
        review_container.empty() 
        with col2:
            agent_app.update_state(config, 
                                   {"execution_result": "Execution blocked by Human Reviewer. The code was deemed unsafe or incorrect."},
                                   as_node="sandbox")
            status_box = st.warning("Execution cancelled. Bypassing to synthesizer...")
            with st.spinner("Formulating fallback response..."):
                for step in agent_app.stream(None, config=config):
                    if "synthesizer" in step:
                        status_box.success("Pipeline completed!")
                        with col1:
                            st.markdown("---")
                            st.subheader("Final Answer")
                            st.write(step["synthesizer"]["final_answer"])