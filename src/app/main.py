import streamlit as st
import os
import sys
from datetime import datetime

# Add project root to sys.path so 'src' can be found
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from src.ai.agents.base_agent import BaseAgent
from src.ai.rag.document_processor import DocumentProcessor
from src.ai.rag.chunker import Chunker
from src.ai.rag.retriever import Retriever # Original default retriever
from src.ai.rag.retrievers import (
    HybridSearchRetriever,
    SemanticQueryExpansionRetriever,
    ContextReRankingRetriever,
    SelfQueryRetriever,
    MultiHopRetriever
)
from src.ai.rag.generator import Generator

st.set_page_config(page_title="Modular RAG", layout="wide")



# Initialize shared components in session state
if "agent" not in st.session_state:
    st.session_state.agent = BaseAgent()
    st.session_state.generator = Generator(st.session_state.agent)
    # Default retriever
    st.session_state.retriever = Retriever(st.session_state.agent)
    st.session_state.strategy = "Default (Semantic)"
    st.session_state.chunks = [] # Store chunks for hybrid search rebuilding

with st.sidebar:
    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("Index Document"):
        import io
        # Buffer the upload
        file_stream = io.BytesIO(uploaded_file.getbuffer())
        filename = uploaded_file.name

        # Workflow
        doc_proc = DocumentProcessor()
        chunker = Chunker()

        status_text = st.empty()
        ingestion_progress = st.progress(0)

        def update_progress(msg, percentage):
            status_text.text(msg)
            ingestion_progress.progress(percentage)

        update_progress("Extracting text (in-memory)...", 10)
        pages = doc_proc.process(file_stream, filename=filename)
        
        update_progress("Splitting text into chunks...", 30)
        chunks = chunker.split_documents(pages)
        st.session_state.chunks = chunks

        # Unified Indexing with detailed callbacks
        def on_index_progress(step_msg):
            # Map internal steps to UI percentages
            if "BM25 index" in step_msg: p = 20
            elif "Saving BM25" in step_msg: p = 40
            elif "Dense Embeddings" in step_msg: p = 60
            elif "Configuring Qdrant" in step_msg: p = 80
            elif "Upserting Dense" in step_msg: p = 95
            else: p = 100
            update_progress(step_msg, p)

        st.session_state.retriever.build_index(
            chunks, 
            on_progress=on_index_progress,
            source=filename,
            date=datetime.now().strftime("%Y-%m-%d")
        )

        update_progress("Indexing complete!", 100)
        import time
        time.sleep(1)
        status_text.empty()
        ingestion_progress.empty()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout: Sidebar (Left) + Main (Middle) + Parameter Panel (Right)
col_main, col_right = st.columns([3, 1], gap="large")

with col_right:
    st.markdown("### Retrieval Strategy")
    strategies = [
        "Default (Semantic)",
        "Hybrid Search",
        "Semantic Query Expansion",
        "Context Re-ranking",
        "Self-Query Retriever",
        "Multi-Hop Retrieval"
    ]
    selected_strategy = st.selectbox(
        "Select Strategy", 
        options=strategies, 
        index=strategies.index(st.session_state.strategy)
    )

    if selected_strategy != st.session_state.strategy:
        st.session_state.strategy = selected_strategy
        # Switch retriever object (this is now instantaneous as indexing is centralized)
        if selected_strategy == "Default (Semantic)":
            st.session_state.retriever = Retriever(st.session_state.agent)
        elif selected_strategy == "Hybrid Search":
            st.session_state.retriever = HybridSearchRetriever(st.session_state.agent)
        elif selected_strategy == "Semantic Query Expansion":
            st.session_state.retriever = SemanticQueryExpansionRetriever(st.session_state.agent)
        elif selected_strategy == "Context Re-ranking":
            st.session_state.retriever = ContextReRankingRetriever(st.session_state.agent)
        elif selected_strategy == "Self-Query Retriever":
            st.session_state.retriever = SelfQueryRetriever(st.session_state.agent)
        elif selected_strategy == "Multi-Hop Retrieval":
            st.session_state.retriever = MultiHopRetriever(st.session_state.agent)
        
        st.rerun()

    st.markdown("### ⚙️ Parameters")
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=3)
    temperature = st.slider("Temperature", min_value=0.0,
                            max_value=1.0, value=0.7, step=0.1)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

with col_main:
    st.title("📄 Local RAG Q&A")
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "reasoning" in msg:
                with st.expander("🔍 Reasoning Step", expanded=False):
                    for i, ctx in enumerate(msg["reasoning"]):
                        st.info(f"**Context Chunk {i+1}**\n\n{ctx}")

    # Chat Input (Sticky at bottom)
    if query := st.chat_input("Ask a question about your document:"):
        # 1. Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Logic & Generation
        if not st.session_state.retriever.has_index:
            with st.chat_message("assistant"):
                err_msg = "Please upload and index a document first."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
        else:
            with st.chat_message("assistant"):
                # Retrieval Reasoning UI
                with st.status(f"🔍 Reasoning Step ({st.session_state.strategy})...", expanded=False) as status:
                    st.write(f"Using {st.session_state.strategy}...")
                    contexts = st.session_state.retriever.retrieve(query, top_k=top_k)
                    st.write(f"Found {len(contexts)} chunks.")
                    for i, ctx in enumerate(contexts):
                        st.info(f"**Context Chunk {i+1}**\n\n{ctx}")
                    status.update(label=f"Reasoning Complete ({len(contexts)} chunks)", state="complete")
                
                # Generation
                with st.spinner("✍️ Writing answer..."):
                    answer = st.session_state.generator.generate_answer(
                        query, contexts, temperature=temperature)
                    st.markdown(answer)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "reasoning": contexts
                })
