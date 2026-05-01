import io
import os
import base64
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def get_drive_service():
    creds_base64 = os.getenv("GOOGLE_CREDS_BASE64")
    
    if creds_base64:
        # Load from environment variable (base64)
        creds_info = json.loads(base64.b64decode(creds_base64).decode('utf-8'))
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=SCOPES
        )
    else:
        # Fallback to local credentials.json
        creds_file = 'credentials.json'
        if os.path.exists(creds_file):
            creds = service_account.Credentials.from_service_account_file(
                creds_file, scopes=SCOPES
            )
        else:
            raise Exception("Google credentials not found (env GOOGLE_CREDS_BASE64 or credentials.json)")

    service = build('drive', 'v3', credentials=creds)
    return service


def download_file(service, file_id, file_name):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)

    # Only download if it's a text file or common document (simplification)
    request = service.files().get_media(fileId=file_id)
    
    with io.FileIO(file_path, 'wb') as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    
    print(f"Downloaded: {file_name}")


def list_and_download_files():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        print("GOOGLE_DRIVE_FOLDER_ID not set. Skipping download.")
        return

    try:
        service = get_drive_service()
        
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=50,
            fields="files(id, name, mimeType)"
        ).execute()

        files = results.get('files', [])

        if not files:
            print("No files found in the specified Google Drive folder.")
            return

        for file in files:
            # We only process text files for this RAG implementation
            if file['mimeType'] == 'text/plain' or file['name'].endswith('.txt'):
                download_file(service, file['id'], file['name'])
            else:
                print(f"Skipping non-text file: {file['name']}")

    except Exception as e:
        print(f"Error syncing with Google Drive: {str(e)}")


if __name__ == "__main__":
    list_and_download_files()