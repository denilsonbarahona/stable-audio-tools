from models import Audio
from generations import generate_audio, process_audio_to_wav_librosa, process_audio_to_wav
from utils.storage import StorageClient
import logging

logger = logging.getLogger(__name__)

class AudioService:
    def __init__(self, storage_client: StorageClient):
        self.storage_client = storage_client

    async def generate_and_store_audio(self, conditioning: Audio, user: str, filename: str) -> dict:
        try:
            audio_name = f"audios/{user}/{filename}.wav"
            audio_tensor = generate_audio(conditioning)
            temp_file = process_audio_to_wav_librosa(audio_tensor)
            audio_url = self.storage_client.upload_file(temp_file, audio_name)
            return {"message": "generated", "url": audio_url}
        except ValueError as e:
            logger.error(f"Error when generate audio: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise