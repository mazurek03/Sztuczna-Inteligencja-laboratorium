import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("samochody")

query = "samochód rodzinny z dużym bagażnikiem"

embedding = embedding_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=3
)

for doc in results["documents"][0]:
    print("=" * 50)
    print(doc)