from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from processing.text_processor import process_file
from embedding.embedder import get_embeddings
from search.vector_store import VectorStore
from llm.generator import generate_answer

import os

app = FastAPI()

vector_store = None


class QueryRequest(BaseModel):
    query: str


def load_data():
    global vector_store

    data_folder = "data"
    vector_store = VectorStore()

    # Ensure data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("Created empty data folder")

    files = os.listdir(data_folder)

    if not files:
        print("No files found in data folder. Running with empty store.")
        return

    for file_name in files:
        file_path = os.path.join(data_folder, file_name)

        chunks = process_file(file_path)
        embeddings = get_embeddings(chunks)

        vector_store.add(embeddings, chunks)

    print("Data loaded successfully")


@app.on_event("startup")
def startup():
    load_data()


@app.post("/ask")
def ask_question(request: QueryRequest):
    global vector_store

    # Handle empty vector store
    if not vector_store or not vector_store.texts:
        return {"answer": "No data available. Please upload documents."}

    query_embedding = get_embeddings([request.query])[0]
    retrieved_chunks = vector_store.search(query_embedding)

    answer = generate_answer(request.query, retrieved_chunks)

    return {"answer": answer}


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
        <head>
            <title>RAG Assistant</title>
            <style>
                body {
                    font-family: Arial;
                    background: #f4f6f8;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    width: 500px;
                }
                h2 {
                    text-align: center;
                }
                input {
                    width: 75%;
                    padding: 10px;
                    margin-top: 10px;
                }
                button {
                    padding: 10px;
                    margin-left: 5px;
                    cursor: pointer;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                #response {
                    margin-top: 20px;
                    background: #f1f1f1;
                    padding: 15px;
                    border-radius: 5px;
                    min-height: 50px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>RAG Assistant</h2>
                <input id="query" placeholder="Ask something..." />
                <button onclick="ask()">Ask</button>

                <div id="response">Your answer will appear here...</div>
            </div>

            <script>
                async function ask() {
                    const query = document.getElementById("query").value;

                    const res = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query })
                    });

                    const data = await res.json();
                    document.getElementById("response").innerText = data.answer;
                }
            </script>
        </body>
    </html>
    """