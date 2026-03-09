from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(r"C:\Users\hvred\Downloads\Introduction-to-AI-and-Basic-Concepts.pdf")

pages = loader.load()

print(len(pages))
print(pages[3])



from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

text = pages[3].page_content

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])

