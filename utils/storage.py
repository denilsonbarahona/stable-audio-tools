from google.cloud import storage
from google.oauth2 import service_account
from utils.config import config
import os
import logging

logger = logging.getLogger(__name__)

class StorageClient:
    def upload_file(self, file_path: str, destination_path: str) -> str:
        raise NotImplementedError("Subclasses must implement upload_file")

class GCSStorageClient(StorageClient):
    def __init__(self, bucket_name: str):
        project_id = config.get_gcp_project_id()
        client_email = config.get_gcp_client_email()
        private_key = config.get_gcp_private_key()

        if project_id and client_email and private_key:
            # Construir credenciales desde variables de entorno
            credentials_info = {
                "type": "service_account",
                "project_id": project_id,
                "client_email": client_email,
                "private_key": private_key,
                "token_uri": "https://oauth2.googleapis.com/token"
            }
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.client = storage.Client(credentials=credentials, project=project_id)
            logger.info("GCS client initialized with env credentials")
        else:
            # Usar ADC si no hay credenciales explícitas (funciona localmente)
            self.client = storage.Client()
            logger.info("GCS client initialized with ADC")
        
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, file_path: str, destination_path: str) -> str:
        logger.info(f"Subiendo a GCS: {destination_path}")
        blob = self.bucket.blob(destination_path)
        blob.upload_from_filename(file_path)
        os.remove(file_path)
        return blob.public_url

BUCKET_NAME = "bondi-ai-audio"
storage_client = GCSStorageClient(BUCKET_NAME)