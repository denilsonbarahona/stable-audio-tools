import runpod
from services.audio_service import AudioService
from utils.storage import storage_client
from adapters.runpod_adapter import event_adapter
from utils.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

runpod.api_key = config.get_runpod_api_key()

audio_service = AudioService(storage_client)

async def handler(event):
    try:
        # Adaptar el evento a datos de la aplicación
        data = event_adapter.extract_data(event)
        user = data["user"]
        filename = data["filename"]
        conditioning = data["conditioning"]

        # Generar y almacenar el audio usando await
        response = await audio_service.generate_and_store_audio(conditioning, user, filename)
        return response
    except Exception as e:
        logger.error(f"Error en el handler: {str(e)}")
        return {"error": str(e)}

# Iniciar el manejador de RunPod
runpod.serverless.start({"handler": handler})