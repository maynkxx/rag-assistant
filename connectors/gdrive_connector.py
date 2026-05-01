from googleapiclient.discovery import build
from processing.text_processor import process_file
from embedding.embedder import get_embeddings
from search.vector_store import VectorStore
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from llm.generator import generate_answer
import io
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'credentials.json'
FOLDER_ID = "1wNNcg5psBpbICJXS4acOJ2VwiGQdmx2o"


def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=creds)
    return service


def download_file(service, file_id, file_name):
    os.makedirs("data", exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    file_path = f"data/{file_name}"

    with io.FileIO(file_path, 'wb') as file:
        downloader = MediaIoBaseDownload(file, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

    print(f"Downloaded: {file_name}")


def list_and_process_files():
    service = get_drive_service()

    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed=false",
        pageSize=10,
        fields="files(id, name, mimeType)"
    ).execute()

    files = results.get('files', [])

    if not files:
        print("No files found.")
        return

    vector_store = None

    for file in files:
        print(f"\nProcessing: {file['name']} ({file['id']})")

        # Step 1: Download
        download_file(service, file['id'], file['name'])

        # Step 2: Process (chunking)
        file_path = f"data/{file['name']}"
        chunks = process_file(file_path)

        print(f"Generated {len(chunks)} chunks")

        # Step 3: Embeddings
        embeddings = get_embeddings(chunks)

        # Step 4: Initialize FAISS once
        if vector_store is None:
            vector_store = VectorStore(dim=len(embeddings[0]))

        # Step 5: Store in FAISS
        vector_store.add(embeddings, chunks)

    # Step 6: Query + Retrieval
    query = "What is refund policy?"
    print(f"\nQuery: {query}")

    query_embedding = get_embeddings([query])[0]
    retrieved_chunks = vector_store.search(query_embedding)

    print("\nRetrieved Context:")
    for r in retrieved_chunks:
        print("-", r)

    # Step 7: Generate final answer
    try:
        answer = generate_answer(query, retrieved_chunks)

        print("\nFinal Answer:")
        print(answer)

    except Exception as e:
        print("\nError generating answer:", str(e))


if __name__ == "__main__":
    list_and_process_files()