### Projekt (Car AI Assistant):
[Projekt_21265](Projekt_21265)  
- Projekt należy uruchomić z pliku *app.py*  
- Niezbędne foldery do działania programu to: *data* oraz *chroma_db*

#### Opis i działanie projektu:
Car AI Assistant to prosty system RAG (Retrieval-Augmented Generation) oparty na lokalnym modelu językowym uruchamianym przez Ollama. Aplikacja pozwala zadawać pytania dotyczące bazy samochodów zapisanej w pliku Excel przy użyciu naturalnego języka.  

#### *Funkcjonalności*  
- Import danych z pliku Excel (.xlsx)
- Lokalny model AI uruchamiany przez Ollama (Llama 3)
- Wyszukiwanie semantyczne z wykorzystaniem ChromaDB i embeddings
- Analiza danych za pomocą Pandas
- Streaming odpowiedzi w czasie rzeczywistym
- Odpowiadanie na pytania (proste i złożone) dotyczące samochodów

#### *Działanie*  
1. Użytkownik wgrywa plik Excel z danymi o samochodach.
2. Dane są przetwarzane i zapisywane w bazie wektorowej ChromaDB.

![](Projekt_21265/1.png)

3. System rozpoznaje typ pytania:  
   - operacje na danych → Pandas (python),
   - rekomendacje i pytania opisowe → RAG + model językowy.  
4. Odpowiedź jest prezentowana w formie konwersacji w aplikacji Streamlit.

![](Projekt/car-rag/Projekt_21265/1)
