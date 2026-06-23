import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("Wczytywanie modelu embeddingów...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

# usuń starą kolekcję jeśli istnieje
try:
    client.delete_collection("samochody")
except:
    pass

collection = client.create_collection("samochody")

df = pd.read_excel("data/auta.xlsx")

for idx, row in df.iterrows():

    document = f"""
Marka: {row['marka']}
Model: {row['model']}
Silnik: {row['silnik']}
Nadwozie: {row['nadwozie']}
Kolor: {row['kolor']}
Miejsca: {row['miejsca']}
Bagażnik: {row['bagaznik_l']} l
Opis: {row['opis']}
"""

    embedding = embedding_model.encode(document).tolist()

    collection.add(
        ids=[str(idx)],
        documents=[document],
        embeddings=[embedding]
    )

print(f"Dodano {len(df)} samochodów do ChromaDB.")