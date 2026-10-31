"""Lab Sheet 5 — Task: "Modify the above app that read more pdfs and gives the
output. Whenever it requires users can upload their own pdfs and get the
answers."

A Streamlit chat app where a user uploads one or more of their own PDFs,
the app indexes them on the spot, and then answers questions grounded in
whatever was uploaded — instead of only reading from a fixed documents/
folder like ingest.py / rag_pipeline.py do.

Usage:
    streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma

from rag_core import load_and_split_pdfs, get_embeddings, build_rag_chain

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on your own uploaded PDF documents")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload one or more PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files and st.button("Process documents"):
    with st.spinner("Reading and indexing your PDFs..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_paths = []
            for uploaded in uploaded_files:
                path = Path(tmp_dir) / uploaded.name
                path.write_bytes(uploaded.getvalue())
                pdf_paths.append(path)

            chunks = load_and_split_pdfs(pdf_paths)

            # No persist_directory: an in-memory store per browser session,
            # so one user's uploads are never mixed with another's.
            vectorstore = Chroma.from_documents(chunks, embedding=get_embeddings())
            st.session_state.chain = build_rag_chain(vectorstore)
            st.session_state.messages = []

    st.success(f"Indexed {len(uploaded_files)} PDF(s) — ask a question below.")

for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

question = st.chat_input("Ask your university question...")

if question:
    if st.session_state.chain is None:
        st.warning("Upload and process at least one PDF first.")
    else:
        st.session_state.messages.append(("user", question))
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)

        st.session_state.messages.append(("assistant", answer))
        st.chat_message("assistant").write(answer)
