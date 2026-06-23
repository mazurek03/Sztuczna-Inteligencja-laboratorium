import streamlit as st
import pandas as pd
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
import tempfile
import re

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Car AI - Agent",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("samochody")

# =========================
# STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "db_built" not in st.session_state:
    st.session_state.db_built = False

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("🚗 Car AI Agent")
    uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])
    
    if st.button("Resetuj czat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# LOAD & CLEAN DATA
# =========================

def build_db(df):
    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])

    for idx, row in df.iterrows():
        doc = f"""
Marka: {row['marka']}
Model: {row['model']}
Silnik: {row['silnik']}
Nadwozie: {row['nadwozie']}
Kolor: {row['kolor']}
Miejsca: {row['miejsca']}
Bagażnik: {row['bagaznik_l']}L
Pojemność skokowa: {row['pojemnosc_skokowa_cm3']} cm3
Średnie spalanie/zużycie: {row['srednie_spalanie']} l lub kWh/100km
Moc (Konie mechaniczne): {row['konie_mechaniczne']} KM
Rodzaj napędu: {row['naped']}
Opis: {row['opis']}
"""
        emb = embedding_model.encode(doc).tolist()
        collection.add(
            ids=[str(idx)],
            documents=[doc],
            embeddings=[emb]
        )

if uploaded_file:
    if st.session_state.df is None or not st.session_state.db_built:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name

        df = pd.read_excel(path)
        
        # Standaryzacja nazw kolumn i tekstu
        df.columns = df.columns.str.strip().str.lower()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            
        st.session_state.df = df
        
        with st.spinner("Buduję bazę danych... To potrwa moment."):
            build_db(df)
            
        st.session_state.db_built = True
        st.success("Baza gotowa!")

if st.session_state.df is None:
    st.warning("Proszę wgrać plik Excel, aby rozpocząć.")
    st.stop()

df = st.session_state.df

# =========================
# HELPER FUNCTIONS
# =========================

def extract_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

def get_text_from_df(target_df):
    docs = []
    for idx, row in target_df.iterrows():
        docs.append(f"Marka: {row['marka']} | Model: {row['model']} | Silnik: {row['silnik']} | Nadwozie: {row['nadwozie']} | Kolor: {row['kolor']} | Miejsca: {row['miejsca']} | Bagażnik: {row['bagaznik_l']}L | Spalanie/Zużycie: {row['srednie_spalanie']} | Moc: {row['konie_mechaniczne']} KM | Napęd: {row['naped']} | Opis: {row['opis']}")
    return "\n\n".join(docs)

# =========================
# AI ROUTER (MÓZG OPERACYJNY)
# =========================

def inteligent_router(q):
    q_low = q.lower()
    
    # PANCERNE WYCIĄGANIE LICZBY ZA POMOCĄ REGEX
    # Szukamy dowolnego ciągu cyfr w tekście
    znalezione_liczby = re.findall(r'\d+', q_low)
    number_trigger = int(znalezione_liczby[0]) if znalezione_liczby else None
    
    # ========================================================
    # PRIORYTET 1: ABSOLUTNE REKORDY (NAJWIĘCEJ, MAX, MIN itp.)
    # ========================================================
    if any(x in q_low for x in ["najwięcej", "najwiecej", "największy", "najwiekszy", "max", "najwyższ", "najwyzsz"]):
        osobowe = df[~df["nadwozie"].str.lower().str.contains("pick|paka", regex=True)]
        
        if "bagaż" in q_low or "bagaz" in q_low:
            top = osobowe.sort_values(by="bagaznik_l", ascending=False).iloc[0]
            return f"Samochodem osobowym o absolutnie największym bagażniku w bazie danych jest **{top['marka']} {top['model']}** z wynikiem **{top['bagaznik_l']} litrów**.", None
        
        if "moc" in q_low or "kon" in q_low or "km" in q_low:
            df["konie_mechaniczne"] = pd.to_numeric(df["konie_mechaniczne"], errors='coerce')
            top = df.sort_values(by="konie_mechaniczne", ascending=False).dropna(subset=["konie_mechaniczne"]).iloc[0]
            return f"Samochodem o największej mocy (KM) w bazie danych jest **{top['marka']} {top['model']}** posiadający aż **{int(top['konie_mechaniczne'])} KM**.", None
            
        if "pojemność" in q_low or "pojemnosc" in q_low or "cm3" in q_low:
            df["pojemnosc_skokowa_cm3"] = pd.to_numeric(df["pojemnosc_skokowa_cm3"], errors='coerce')
            top = df.sort_values(by="pojemnosc_skokowa_cm3", ascending=False).dropna(subset=["pojemnosc_skokowa_cm3"]).iloc[0]
            return f"Samochodem o największej pojemności skokowej silnika jest **{top['marka']} {top['model']}** – **{int(top['pojemnosc_skokowa_cm3'])} cm3**.", None

    if any(x in q_low for x in ["najmniejsze", "najniższe", "najniezsz", "min", "najmniej"]):
        if "spalanie" in q_low or "zużycie" in q_low or "zuzycie" in q_low:
            df["srednie_spalanie"] = pd.to_numeric(df["srednie_spalanie"], errors='coerce')
            if "elektryk" in q_low or "elektrycz" in q_low:
                spalinowe = df[df["silnik"].str.lower().str.contains("elektryczny|ev")]
                jednostka = "kWh/100km"
            else:
                spalinowe = df[~df["silnik"].str.lower().str.contains("elektryczny|ev")]
                jednostka = "l/100km"
            
            top = spalinowe.sort_values(by="srednie_spalanie", ascending=True).dropna(subset=["srednie_spalanie"]).iloc[0]
            return f"Samochodem o najniższym zużyciu/spalaniu w tej kategorii jest **{top['marka']} {top['model']}** z wynikiem **{top['srednie_spalanie']} {jednostka}**.", None

    # ========================================================
    # PRIORYTET 2: FILTRY WARUNKOWE (POWYŻEJ, PONIŻEJ, >, <) - DLA "ILE" ORAZ "KTÓRE"
    # ========================================================
    if number_trigger is not None:
        is_above = any(x in q_low for x in ["powyżej", "powyzej", "więcej niż", "wiecej niz", "wieksze", "większe", ">"])
        is_below = any(x in q_low for x in ["poniżej", "ponizej", "mniej niż", "mniej niz", "mniejsze", "<"])
        
        if is_above or is_below:
            target_col = None
            col_label = ""
            unit = ""
            
            # 1. SPALANIE / ZUŻYCIE MA TERAZ NAJWYŻSZY PRIORYTET
            if "spalanie" in q_low or "zużycie" in q_low or "zuzycie" in q_low or "pali" in q_low:
                target_col = "srednie_spalanie"
                col_label = "spalania/zużycia"
                unit = "l lub kWh/100km"
            
            # 2. POJEMNOŚĆ SILNIKA
            elif "cm3" in q_low or "pojemność" in q_low or "pojemnosc" in q_low or "pojemnosci" in q_low:
                target_col = "pojemnosc_skokowa_cm3"
                col_label = "pojemności skokowej"
                unit = "cm3"
                
            # 3. MOC (KM)
            elif "km" in q_low or "moc" in q_low or "kon" in q_low:
                target_col = "konie_mechaniczne"
                col_label = "mocy"
                unit = "KM"
                
            # 4. BAGAŻNIK
            elif "bagaż" in q_low or "bagaz" in q_low or "bagażnika" in q_low or "bagaznika" in q_low:
                target_col = "bagaznik_l"
                col_label = "pojemności bagażnika"
                unit = "L"
                
            # 5. INTELIGENTNY DETEKTOR SŁOWA "LITR"
            elif "litr" in q_low:
                # Jeśli użytkownik podał "litrów", ale nie napisał wprost "spalanie" ani "bagażnik":
                if number_trigger > 50:
                    target_col = "bagaznik_l"
                    col_label = "pojemności bagażnika"
                    unit = "L"
                else:
                    target_col = "srednie_spalanie"
                    col_label = "spalania/zużycia"
                    unit = "l/100km"

            # 6. AWARYJNY BEZPIECZNIK (brak słów kluczowych, sama liczba)
            if not target_col:
                if number_trigger > 500:
                    target_col = "pojemnosc_skokowa_cm3"
                    col_label = "pojemności skokowej"
                    unit = "cm3"
                elif 50 <= number_trigger <= 500:
                    target_col = "konie_mechaniczne"
                    col_label = "mocy"
                    unit = "KM"
                else:
                    target_col = "srednie_spalanie"
                    col_label = "spalania/zużycia"
                    unit = "l/100km"

            # ========================================================
            # DALSZA CZĘŚĆ KODU (RZUTOWANIE I FILTROWANIE DATAFRAME)
            # ========================================================
            if target_col:
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                
                if is_above:
                    pasujace_auta = df[df[target_col] > number_trigger]
                    relacja = "powyżej"
                else:
                    pasujace_auta = df[df[target_col] < number_trigger]
                    if target_col == "pojemnosc_skokowa_cm3":
                        pasujace_auta = pasujace_auta[pasujace_auta[target_col] > 0]
                    relacja = "poniżej"
                
                if "ile" in q_low:
                    return f"W bazie danych znajduje się dokładnie **{len(pasujace_auta)}** samochodów o wartości {col_label} {relacja} {number_trigger} {unit}.", None
                
                if pasujace_auta.empty:
                    return f"W bazie danych nie ma samochodów o wartości {col_label} {relacja} {number_trigger} {unit}.", None
                
                # Dynamiczne wyświetlanie wartości zmiennoprzecinkowych dla spalania
                lista_aut = []
                for _, r in pasujace_auta.iterrows():
                    val = r[target_col]
                    # Formatujemy jako int dla pojemności/mocy/bagażnika, zostawiamy float dla spalania
                    val_str = f"{val}" if target_col == "srednie_spalanie" else f"{int(val) if not pd.isna(val) else val}"
                    lista_aut.append(f"- **{r['marka']} {r['model']}** ({val_str} {unit} | {r['silnik']}, {r['nadwozie']})")
                
                lista_aut_str = "\n".join(lista_aut)
                return f"Samochody o parametrach {col_label} **{relacja} {number_trigger} {unit}** to:\n{lista_aut_str}", None
            
    # ========================================================
    # PRIORYTET 3: STANDARDOWE ZAPYTANIA O ILOŚĆ ("ILE...")
    # ========================================================
    if "ile" in q_low:
        # A. Liczenie po napędzie (Kluczowa poprawka!)
        if "rwd" in q_low or "tył" in q_low or "tyln" in q_low:
            count = len(df[df["naped"].str.lower().str.contains("rwd|tył|tylny", regex=True)])
            return f"W bazie danych znajduje się dokładnie **{count}** samochodów z napędem na tylną oś (RWD).", None
            
        if "awd" in q_low or "4x4" in q_low or "cztery" in q_low or "4wd" in q_low:
            count = len(df[df["naped"].str.lower().str.contains("awd|4x4|4wd", regex=True)])
            return f"W bazie danych znajduje się dokładnie **{count}** samochodów z napędem na cztery koła (AWD/4x4).", None
            
        if "fwd" in q_low or "przód" in q_low or "przedn" in q_low:
            count = len(df[df["naped"].str.lower().str.contains("fwd|przód|przedni", regex=True)])
            return f"W bazie danych znajduje się dokładnie **{count}** samochodów z napędem na przednią oś (FWD).", None

        # B. Liczenie po kolorach
        dostepne_kolory = df["kolor"].dropna().str.lower().unique()
        for k in dostepne_kolory:
            if k[:4] in q_low:
                count = len(df[df["kolor"].str.lower() == k])
                return f"W bazie danych znajduje się dokładnie **{count}** samochodów o kolorze {k}.", None
        
        # C. Liczenie po nadwoziach
        dostepne_nadwozia = df["nadwozie"].dropna().str.lower().unique()
        for n in dostepne_nadwozia:
            if n in q_low:
                count = len(df[df["nadwozie"].str.lower() == n])
                return f"W bazie danych znajduje się dokładnie **{count}** samochodów o nadwoziu {n}.", None

        # D. Liczenie po typie silnika
        if "diesel" in q_low or "rop" in q_low:
            count = len(df[df["silnik"].str.lower().str.contains("diesel")])
            return f"W bazie danych znajduje się dokładnie **{count}** samochodów z silnikiem Diesla.", None
        if "elektryk" in q_low or "elektrycz" in q_low:
            count = len(df[df["silnik"].str.lower().str.contains("elektryczny|ev")])
            return f"W bazie danych znajduje się dokładnie **{count}** samochodów z silnikiem w 100% elektrycznym.", None
        if "benzyn" in q_low:
            count = len(df[df["silnik"].str.lower().str.contains("benzyna|benzynowy")])
            return f"W bazie danych znajduje się dokładnie **{count}** samochodów z silnikiem benzynowym.", None
        
    # ========================================================
    # PRIORYTET 4: STANDARDOWE ZAPYTANIA O LISTĘ / CZY JEST ("KTÓRE...", "CZY JEST...")
    # ========================================================
    if any(x in q_low for x in ["które", "ktore", "wypisz", "pokaż", "pokaz", "czy jest", "jakie"]):
        
        # KROK 1: NAJPIERW SPRAWDZAMY CZY UŻYTKOWNIK PYTA O KONKRETNĄ MARKĘ
        dostepne_marki = df["marka"].dropna().str.lower().unique()
        for m in dostepne_marki:
            if m in q_low:
                # Wycinamy z bazy TYLKO auta tej konkretnej marki (np. Nissan)
                auta_marki = df[df["marka"].str.lower() == m]
                
                # Czy użytkownik pyta w tym samym zdaniu o silnik elektryczny/diesel dla tej marki?
                if "elektryk" in q_low or "elektrycz" in q_low:
                    auta_marki = auta_marki[auta_marki["silnik"].str.lower().str.contains("elektryczny|ev")]
                    if auta_marki.empty:
                        return f"W bazie danych posiadamy modele marki **{m.capitalize()}**, ale żaden z nich **nie jest samochodem elektrycznym**.", None
                    lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['nadwozie']}, Kolor: {r['kolor']})" for _, r in auta_marki.iterrows()])
                    return f"Elektryczne modele marki **{m.capitalize()}** w bazie to:\n{lista_aut}", None
                
                if "diesel" in q_low or "rop" in q_low:
                    auta_marki = auta_marki[auta_marki["silnik"].str.lower().str.contains("diesel")]
                    if auta_marki.empty:
                        return f"W bazie danych posiadamy modele marki **{m.capitalize()}**, ale żaden z nich **nie ma silnika Diesla**.", None
                    lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['nadwozie']}, Kolor: {r['kolor']})" for _, r in auta_marki.iterrows()])
                    return f"Modele marki **{m.capitalize()}** z silnikiem Diesla to:\n{lista_aut}", None

                # Czy użytkownik pyta w tym samym zdaniu o nadwozie dla tej marki?
                dostepne_nadwozia = df["nadwozie"].dropna().str.lower().unique()
                for n in dostepne_nadwozia:
                    if n in q_low:
                        auta_marki = auta_marki[auta_marki["nadwozie"].str.lower() == n]
                        if auta_marki.empty:
                            return f"W bazie danych posiadamy markę **{m.capitalize()}**, ale żaden model nie występuje w nadwoziu **{n}**.", None
                        lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['silnik']}, Kolor: {r['kolor']})" for _, r in auta_marki.iterrows()])
                        return f"Modele marki **{m.capitalize()}** o nadwoziu **{n}** w naszej bazie to:\n{lista_aut}", None
                
                # Czy użytkownik pyta ogólnie o kolory danej marki?
                if "kolor" in q_low:
                    kolory_marki = auta_marki["kolor"].unique()
                    lista_kolorow = ", ".join([str(k) for k in kolory_marki])
                    lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** (Kolor: {r['kolor']})" for _, r in auta_marki.iterrows()])
                    return f"Modele marki **{m.capitalize()}** są dostępne w następujących kolorach: {lista_kolorow}.\nOto pełna lista wykazanych aut:\n{lista_aut}", None

                # Jeśli pytał po prostu ogólnie o markę (np. "Jakie masz Nissany?")
                lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['silnik']}, {r['nadwozie']}, Kolor: {r['kolor']})" for _, r in auta_marki.iterrows()])
                return f"Samochody marki **{m.capitalize()}** dostępne w bazie danych to:\n{lista_aut}", None

        # KROK 2: DOPIERO JEŚLI W PYTANIU NIE MA MARKI, OBSŁUGUJEMY OGÓLNE FILTRY SILNIKÓW
        if "diesel" in q_low or "rop" in q_low:
            pasujace_auta = df[df["silnik"].str.lower().str.contains("diesel")]
            lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['nadwozie']}, Kolor: {r['kolor']})" for _, r in pasujace_auta.iterrows()])
            return f"Samochody w bazie z silnikiem **Diesla** to:\n{lista_aut}", None
            
        if "elektryk" in q_low or "elektrycz" in q_low:
            pasujace_auta = df[df["silnik"].str.lower().str.contains("elektryczny|ev")]
            lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['nadwozie']}, Kolor: {r['kolor']})" for _, r in pasujace_auta.iterrows()])
            return f"Samochody w bazie z silnikiem **Elektrycznym** to:\n{lista_aut}", None

        if "benzyn" in q_low:
            pasujace_auta = df[df["silnik"].str.lower().str.contains("benzyna|benzynowy")]
            lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['nadwozie']}, Kolor: {r['kolor']})" for _, r in pasujace_auta.iterrows()])
            return f"Samochody w bazie z silnikiem **Benzynowym** to:\n{lista_aut}", None

        # KROK 3: POZOSTAŁE OGÓLNE FILTRY (SAM KOLOR, SAMO NADWOZIE)
        dostepne_kolory = df["kolor"].dropna().str.lower().unique()
        for k in dostepne_kolory:
            if k[:4] in q_low:
                pasujace_auta = df[df["kolor"].str.lower() == k]
                if not pasujace_auta.empty:
                    lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['silnik']}, {r['nadwozie']})" for _, r in pasujace_auta.iterrows()])
                    return f"Samochody w bazie o kolorze **{k}** to:\n{lista_aut}", None

        dostepne_nadwozia = df["nadwozie"].dropna().str.lower().unique()
        for n in dostepne_nadwozia:
            if n in q_low:
                pasujace_auta = df[df["nadwozie"].str.lower() == n]
                if not pasujace_auta.empty:
                    lista_aut = "\n".join([f"- **{r['marka']} {r['model']}** ({r['silnik']}, Kolor: {r['kolor']})" for _, r in pasujace_auta.iterrows()])
                    return f"Samochody w bazie o nadwoziu **{n}** to:\n{lista_aut}", None
                
    # ========================================================
    # PRIORYTET 5: KROK RODZINNY (Miejsca)
    # ========================================================
    if number_trigger and ("osób" in q_low or "osob" in q_low or "miejsc" in q_low or "rodzin" in q_low):
        odpowiednie_auta = df[df["miejsca"] >= number_trigger]
        if odpowiednie_auta.empty:
            return f"Niestety, w bazie danych nie ma żadnego samochodu, który pomieści przynajmniej {number_trigger} osób.", None
        context_miejsca = get_text_from_df(odpowiednie_auta.head(7))
        return None, context_miejsca

    # ========================================================
    # PRIORYTET 6: KONTROLA FAKTÓW PRZED RAG-iem
    # ========================================================
    rag_df = df.copy()
    if "rwd" in q_low or "tył" in q_low or "tyln" in q_low:
        rag_df = rag_df[rag_df["naped"].str.lower().str.contains("rwd|tył|tylny", regex=True)]
    elif "awd" in q_low or "4x4" in q_low or "cztery" in q_low:
        rag_df = rag_df[rag_df["naped"].str.lower().str.contains("awd|4x4|4wd", regex=True)]
    elif "fwd" in q_low or "przód" in q_low or "przedn" in q_low:
        rag_df = rag_df[rag_df["naped"].str.lower().str.contains("fwd|przód|przedni", regex=True)]

    dostepne_nadwozia = rag_df["nadwozie"].dropna().str.lower().unique()
    for n in dostepne_nadwozia:
        if n in q_low:
            rag_df = rag_df[rag_df["nadwozie"].str.lower() == n]
            break

    if 0 < len(rag_df) < len(df):
        context_przefiltrowany = get_text_from_df(rag_df.head(6))
        return None, context_przefiltrowany

    # ========================================================
    # PRIORYTET 7: REZERWOWY STANDARDOWY RAG SEMANTYCZNY
    # ========================================================
    emb = embedding_model.encode(q).tolist()
    res = collection.query(query_embeddings=[emb], n_results=7)
    return None, "\n\n".join(res["documents"][0])


# =========================
# STREAM RESPONSE
# =========================

def stream_llm(messages):
    response = ollama.chat(
        model="llama3",
        messages=messages,
        stream=True,
        options={"temperature": 0.1}
    )
    full = ""
    placeholder = st.empty()
    for chunk in response:
        if "message" in chunk and "content" in chunk["message"]:
            full += chunk["message"]["content"]
            placeholder.markdown(f"🤖 AI:\n\n{full}")
    return full

# =========================
# UI CHAT
# =========================

st.title("🚗 Zaawansowany Doradca Car AI (Tryb Agentowy)")

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"**👤 Ty:** {m['content']}")
    else:
        st.markdown(f"**🤖 AI:** {m['content']}")

user_input = st.chat_input("Zadaj pytanie o samochody z bazy...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"**👤 Ty:** {user_input}")

    # Wywołanie inteligentnego podziału zadań
    direct_answer, rag_context = inteligent_router(user_input)

    if direct_answer:
        # Python wyliczył to sam! Wyświetlamy od razu 100% poprawny wynik.
        st.markdown(f"🤖 AI:\n\n{direct_answer}")
        st.session_state.messages.append({"role": "assistant", "content": direct_answer})
    else:
        # Pytanie opisowe/doradcze -> Przekazujemy wyselekcjonowane dane do Llamy 3
        messages = [
            {
                "role": "system",
                "content": """Jesteś ekspertem motoryzacyjnym CarAI. Odpowiadasz logicznie, profesjonalnie i wyłącznie w języku polskim na podstawie dostarczonych danych.
                
Zasady:
1. Odpowiadaj konkretnie na pytanie użytkownika. Nie zmyślaj parametrów.
2. Jeśli użytkownik prosi o porównanie lub wybór, przeanalizuj cechy podanych aut i uzasadnij swój wybór.
3. Pisz ładną, naturalną polszczyzną, unikaj zwrotów technicznych z instrukcji systemowych."""
            },
            {
                "role": "user",
                "content": f"""DANE O SAMOCHODACH:
{rag_context}

PYTANIE UŻYTKOWNIKA:
{user_input}

ZAKAZ: Pod żadnym pozorem nie używaj języka angielskiego. Odpowiedz krótko i wyłącznie po polsku."""
            }
        ]
        
        answer = stream_llm(messages)
        if answer is None:
            answer = ""
        st.session_state.messages.append({"role": "assistant", "content": answer})