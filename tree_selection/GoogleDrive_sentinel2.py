import io
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import Flow

# Function to authenticate and create a service
def create_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    #CLIENT_SECRETS_FILE = ''

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob')

    auth_url, _ = flow.authorization_url(prompt='consent')
    print('Please go to this URL and authorize access:', auth_url)
    code = input('Enter the authorization code here: ')
    flow.fetch_token(code=code)

    credentials = flow.credentials
    service = build('drive', 'v3', credentials=credentials)
    return service

def find_folders_with_prefix(service, prefix='Sentinel2'):
    """Find all folders starting with a specific prefix."""
    results = []
    page_token = None
    while True:
        response = service.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and name contains '{prefix}'",
            spaces='drive',
            fields='nextPageToken, files(id, name)',
            pageToken=page_token
        ).execute()
        results.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return results

def download_folder_contents(service, folder_id, folder_name, base_path):
    """Recursively download the contents of a folder."""
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token
        ).execute()

        items = response.get('files', [])
        for item in items:
            file_path = os.path.join(folder_path, item['name'])
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                download_folder_contents(service, item['id'], item['name'], folder_path)
            else:
                if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
                    print(f"Skipping already downloaded file: {item['name']}")
                    continue
                try:
                    request = service.files().get_media(fileId=item['id'])
                    with io.FileIO(file_path, 'wb') as fh:
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                            print(f"Downloading {item['name']}: {int(status.progress() * 100)}%")
                except HttpError as error:
                    print(f"Failed to download {item['name']}: {error}")

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

def main():
    service = create_service()
    base_path = './Sentinel_gbg'
    folders = find_folders_with_prefix(service, 'Sentinel2')
    for folder in folders:
        print(f"Downloading contents of folder: {folder['name']}")
        download_folder_contents(service, folder['id'], folder['name'], base_path)

if __name__ == "__main__":
    main()
