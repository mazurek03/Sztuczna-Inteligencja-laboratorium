import pandas as pd
import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# =====================
# Wczytanie danych
# =====================

df = pd.read_excel("data/auta.xlsx")

# =====================
# ChromaDB
# =====================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "samochody"
)

# =====================
# Funkcje Pandas
# =====================

def count_black_cars():
    return len(df[df["kolor"].str.lower() == "czarny"])

def list_diesel_cars():

    diesels = df[
        df["silnik"].str.lower() == "diesel"
    ]

    result = []

    for _, row in diesels.iterrows():
        result.append(
            f"{row['marka']} {row['model']}"
        )

    return "\n".join(result)

def biggest_trunk():

    max_size = df["bagaznik_l"].max()

    cars = df[
        df["bagaznik_l"] == max_size
    ]

    result = []

    for _, row in cars.iterrows():
        result.append(
            f"{row['marka']} {row['model']} ({max_size} l)"
        )

    return "\n".join(result)

def brands():

    return ", ".join(
        sorted(df["marka"].unique())
    )

# =====================
# RAG
# =====================

def rag_answer(question):

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
Jesteś ekspertem motoryzacyjnym.

Odpowiadaj wyłącznie na podstawie
dostarczonych danych.

Jeżeli dane nie pozwalają
jednoznacznie odpowiedzieć,
napisz o tym.

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

    return response["message"]["content"]

# =====================
# Chat
# =====================

while True:

    question = input("\nPytanie: ")

    q = question.lower()

    if q in ["exit", "quit", "koniec"]:
        break

    # ---------
    # Pandas
    # ---------

    if "czarn" in q and "ile" in q:

        answer = (
            f"W bazie znajduje się "
            f"{count_black_cars()} "
            f"czarnych samochodów."
        )

    elif "diesel" in q:

        answer = (
            "Samochody z silnikiem Diesel:\n\n"
            + list_diesel_cars()
        )

    elif "największy bagażnik" in q:

        answer = (
            "Największy bagażnik mają:\n\n"
            + biggest_trunk()
        )

    elif "marki" in q:

        answer = (
            "Marki w bazie:\n\n"
            + brands()
        )

    # ---------
    # RAG
    # ---------

    else:

        answer = rag_answer(question)

    print("\nODPOWIEDŹ:\n")
    print(answer)