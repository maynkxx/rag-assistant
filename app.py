import os
import json
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from connectors.gdrive_connector import list_and_download_files
from processing.text_processor import process_file
from embedding.embedder import get_embeddings
from search.vector_store import VectorStore
from llm.generator import generate_answer

load_dotenv()

# Global vector store instance
vector_store = VectorStore()


class QueryRequest(BaseModel):
    query: str


def sync_and_load():
    global vector_store
    
    print("🚀 Starting sync with Google Drive...")
    list_and_download_files()
    
    print("📂 Processing downloaded files...")
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    
    if not files:
        print("⚠️ No text files found to process.")
        return

    # Clear existing store to avoid duplicates on re-sync
    vector_store = VectorStore()

    for file_name in files:
        file_path = os.path.join(data_folder, file_name)
        print(f"📄 Processing {file_name}...")
        try:
            chunks = process_file(file_path)
            if chunks:
                embeddings = get_embeddings(chunks)
                vector_store.add(embeddings, chunks)
        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")

    print(f"✅ Data loaded successfully. Total chunks: {len(vector_store.texts)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    sync_and_load()
    yield

app = FastAPI(title="RAG GDrive Assistant", lifespan=lifespan)


@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not vector_store.texts:
        return {"answer": "I don't have any data yet. Please ensure files are in the Google Drive folder."}

    try:
        # Get query embedding
        query_embeddings = get_embeddings([request.query])
        query_embedding = query_embeddings[0]
        
        # Retrieve relevant chunks
        retrieved_chunks = vector_store.search(query_embedding)
        
        if not retrieved_chunks:
            return {"answer": "I couldn't find any relevant information in the documents."}
            
        # Generate answer using LLM
        answer = generate_answer(request.query, retrieved_chunks)
        return {"answer": answer}
        
    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Knowledge Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text: #f8fafc;
                --text-muted: #94a3b8;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                overflow: hidden;
            }
            .container {
                background: var(--card-bg);
                padding: 2.5rem;
                border-radius: 1.5rem;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                width: 100%;
                max-width: 600px;
                transition: transform 0.3s ease;
            }
            h1 {
                font-weight: 600;
                margin-bottom: 0.5rem;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
            }
            p.subtitle {
                color: var(--text-muted);
                text-align: center;
                margin-bottom: 2rem;
                font-size: 0.9rem;
            }
            .input-group {
                display: flex;
                gap: 0.5rem;
                margin-bottom: 1.5rem;
            }
            input {
                flex: 1;
                background: #334155;
                border: 1px solid #475569;
                border-radius: 0.75rem;
                padding: 1rem;
                color: white;
                font-size: 1rem;
                outline: none;
                transition: border-color 0.2s;
            }
            input:focus {
                border-color: var(--primary);
            }
            button {
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 0.75rem;
                padding: 0 1.5rem;
                font-weight: 600;
                cursor: pointer;
                transition: filter 0.2s;
            }
            button:hover {
                filter: brightness(1.1);
            }
            #response {
                background: #0f172a;
                border-radius: 1rem;
                padding: 1.5rem;
                min-height: 100px;
                max-height: 300px;
                overflow-y: auto;
                line-height: 1.6;
                font-size: 0.95rem;
                border-left: 4px solid var(--primary);
            }
            .loading {
                color: var(--text-muted);
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Knowledge AI</h1>
            <p class="subtitle">Syncing GDrive Docs in Real-time</p>
            
            <div class="input-group">
                <input type="text" id="query" placeholder="Ask a question about your documents..." onkeypress="if(event.key==='Enter') ask()">
                <button onclick="ask()">Ask</button>
            </div>

            <div id="response">Your AI-generated answer will appear here...</div>
        </div>

        <script>
            async function ask() {
                const queryInput = document.getElementById("query");
                const responseDiv = document.getElementById("response");
                const query = queryInput.value.trim();
                
                if (!query) return;

                responseDiv.innerHTML = '<span class="loading">Thinking...</span>';
                queryInput.disabled = true;

                try {
                    const res = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query })
                    });

                    const data = await res.json();
                    responseDiv.innerText = data.answer;
                } catch (err) {
                    responseDiv.innerText = "Error: " + err.message;
                } finally {
                    queryInput.disabled = false;
                    queryInput.focus();
                }
            }
        </script>
    </body>
    </html>
    """