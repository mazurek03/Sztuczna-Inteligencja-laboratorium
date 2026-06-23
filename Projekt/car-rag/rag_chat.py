import chromadb
import ollama

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "samochody"
)

while True:

    question = input("\nPytanie: ")

    if question.lower() in ["exit", "quit", "koniec"]:
        break

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n\n".join(
        results["documents"][0]
    )

    prompt = f"""
Jesteś asystentem analizującym bazę samochodów.

Odpowiadaj wyłącznie na podstawie
dostarczonych danych.

Jeżeli nie ma informacji w danych,
napisz że nie znaleziono informacji.

DANE:

{context}

PYTANIE:

{question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nODPOWIEDŹ:\n")
    print(
        response["message"]["content"]
    )