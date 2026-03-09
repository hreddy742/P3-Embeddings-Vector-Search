from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embed  = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_texts(
    texts=["I love dogs", "I adore puppies", "Tax filing is due"],
    embedding = embed
)

results = db.similarity_search('pets and animals', k=2)

for r in results:
    print(r.page_content)
    print(r.metadata)
    print("\n")