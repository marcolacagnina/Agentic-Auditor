import streamlit as st
import time
from src.production.graph import app as agent_app

# Configurazione della pagina
st.set_page_config(page_title="AI Auditor", layout="wide")

st.title("Agentic Auditor")
st.markdown("""
This dashboard shows a **Hybrid Edge-cloud** architecture.
- The docs do not exit from the server (local RAG).
- The reasoning and the Python code are locally generated (Qwen 3B Coder) for privacy.
- The math execution takes place in a separated sandbox.
""")

with st.sidebar:
    st.header("🧠 Agent Architecture")
    st.markdown("Live DAG (Directed Acyclic Graph) state.")
    
    try:
        raw_mermaid = agent_app.get_graph().draw_mermaid()
        
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
        
        styled_mermaid = raw_mermaid + "\n" + custom_styles
        
        st.markdown(f"```mermaid\n{styled_mermaid}\n```")
        
        st.markdown("""
        **Legend:**
        - 🔵 Cloud Reasoning (Groq)
        - 🟠 Vector DB (Chroma)
        - 🟣 Local Code Gen (MLX)
        - 🔴 Execution Sandbox
        """)
        
    except Exception as e:
        st.warning(f"Error: {e}")


col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Chat")
    user_query = st.text_area("Ask a question:", 
                              value="Calculate the percentage increase in Apple's Total Revenue from the fiscal year ending 2024-09-30 to the fiscal year ending 2025-09-30.")
    
    submit = st.button("Run", type="primary")

if submit:
    with col2:
        st.subheader("LangGraph Execution Trace")
        
        # Placeholder for updating in real time
        status_box = st.info("Initializing Agent State...")
        
        # === Retrieve ===
        time.sleep(2) # Delay for visual effect
        status_box.info("[Node: Retrieve] Fetching context from local ChromaDB...")
        state = {"question": user_query}
        
        try:
            # Execute the graph step by step
            for step in agent_app.stream(state):
                
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
                    status_box.warning("[Node: MLX Coder] Generating reasoning and Python code locally...")
                    with st.expander("Show Internal Thought Process", expanded=True):
                        st.markdown(f"**Thought:**\n{step['coder']['thought']}")
                        st.code(step['coder']['python_code'], language='python')
                        
                elif "sandbox" in step:
                    status_box.error("[Node: Sandbox] Executing generated Python code...")
                    with st.expander("Show Sandbox Output", expanded=True):
                        st.text(step['sandbox']['execution_result'])
                        
                elif "synthesizer" in step:
                    status_box.info("[Node: Synthesizer] Formulating final response...")
                    
            status_box.success("Pipeline completed!")
            
            # Show final answer in the left column
            with col1:
                st.markdown("---")
                st.subheader("Final Answer")
                st.write(step["synthesizer"]["final_answer"])
                
        except Exception as e:
            st.error(f"Pipeline Error: {str(e)}")