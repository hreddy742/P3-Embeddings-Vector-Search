# app.py
# P3 - Baby RAG - Streamlit UI
# Upload a PDF, ask questions, get answers with page citations

import tempfile
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Baby RAG", page_icon="📄")
st.title("📄 Baby RAG — Ask Your PDF")
st.write("Upload a PDF and ask any question about it.")

# ── INITIALISE MODELS ─────────────────────────────────────────────────────
# @st.cache_resource creates these once and reuses them
# without this - new model object created on every interaction

@st.cache_resource
def load_models():
    embed = OllamaEmbeddings(model="nomic-embed-text")
    llm   = ChatOllama(model="llama3.1:8b", temperature=0)
    return embed, llm

embed, llm = load_models()

# ── PROMPT ────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer the question using ONLY the context provided.
If the answer is not in the context say 'I could not find this in the document'."""),
    ("human", """Context:
{context}

Question: {question}""")
])

chain = prompt | llm

# ── FILE UPLOAD ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is None:
    # nothing uploaded yet - show instructions and stop
    st.info("Please upload a PDF to get started.")
    st.stop()

# ── BUILD VECTORSTORE ─────────────────────────────────────────────────────
# st.cache_resource with the filename as key
# means: if same PDF uploaded again - use cached vectorstore
# if new PDF uploaded - rebuild from scratch

@st.cache_resource
def build_vectorstore(filename):
    # save uploaded file to disk temporarily
    # PyPDFLoader needs a real file path not a file object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # load and split
    loader   = PyPDFLoader(tmp_path)
    pages    = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(pages)

    # embed and store
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embed
    )

    # clean up temp file
    os.unlink(tmp_path)

    return vectorstore, len(pages), len(chunks)

# show spinner while building
with st.spinner("Reading and indexing your PDF... please wait"):
    vectorstore, num_pages, num_chunks = build_vectorstore(uploaded_file.name)

# show PDF info
st.success(f"Ready — {num_pages} pages, {num_chunks} chunks indexed")

# ── QUESTION AND ANSWER ───────────────────────────────────────────────────
st.divider()

question = st.text_input("Ask a question about your document:")

if question:

    with st.spinner("Searching and generating answer..."):

        # retrieve relevant chunks
        relevant_chunks = vectorstore.similarity_search(question, k=3)

        # get page numbers
        pages_used = list(set([
            chunk.metadata.get("page", "unknown")
            for chunk in relevant_chunks
        ]))

        # build context
        context = "\n\n".join([c.page_content for c in relevant_chunks])

        # get answer
        response = chain.invoke({
            "context":  context,
            "question": question
        })

    # show answer
    st.markdown("### Answer")
    st.write(response.content)

    # show sources
    st.markdown("### Sources")
    st.write(f"Answer found in pages: {sorted(pages_used)}")

    # show the actual chunks used - in expandable section
    with st.expander("See retrieved chunks"):
        for i, chunk in enumerate(relevant_chunks, 1):
            page = chunk.metadata.get("page", "unknown")
            st.markdown(f"**Chunk {i} — Page {page}**")
            st.write(chunk.page_content)
            st.divider()
