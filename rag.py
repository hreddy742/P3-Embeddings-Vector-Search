from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

PDF_PATH = "Introduction-to-AI-and-Basic-Concepts.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(pages)

embed = OllamaEmbeddings(model = 'nomic-embed-text')

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embed
)

llm = ChatOllama(model = 'llama3.1:8b')

prompt = ChatPromptTemplate.from_messages([
    ('system',"""You are a helpgul assistant.
    Answer the question using ONLY the context provided below.
If the answer is not in the context say 'I could not find this in the document'."""),
('human',"""Context: {context}

Question: {question}""")
])

chain = prompt | llm

questions = [
    "What is this document about?",
    "What does it say about artificial intelligence?",
    "What does it say about data?"
]

for q in questions:
    retrieved = vectorstore.similarity_search(q, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    response = chain.invoke({"context": context, "question": q})
    print(f"Question: {q}")
    print(f"Answer: {response.content}")
    print("\n")