from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

PDF_PATH = "VR-001-BK.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)


chunk = splitter.split_documents(pages)

embed = OllamaEmbeddings(model = 'nomic-embed-text')

vectorstore = Chroma.from_documents(chunk,embed)

llm = ChatOllama(model = 'llama3.1:8b',streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ('system','You are the ramayan Pandit.Answer the question using the contect provided only.If you do not know Say I do not know.'),
    ('human','Context:{context}\n\nQuestion:{question}') 
])

chain = prompt | llm
print('You ramayan pandit is ready to answer your question')
print('Type """exit""" to quit the chat')

greetings = {'hi', 'hello', 'hey', 'hii', 'namaste'}

while True:
    question = input('you: ')
    if question.lower().strip() in  {'exit','quit','bye'}:
        break
    if question.lower() in greetings:
        print("Answer: Hello Harsha! Ask me anything about the Ramayan.\n")
        continue
    retrieved = vectorstore.similarity_search(question,k=3)
    pages_used=[]
    for chunk in retrieved:
        pages_used.append(chunk.metadata.get('page','unknown'))
    context = "\n\n".join([doc.page_content for doc in retrieved])

    for chunk in chain.stream({
    "context": context,
    "question": question
    }):
        print(chunk.content, end="", flush=True)

    print("\n")
    print(f"Sources: pages {pages_used}\n")