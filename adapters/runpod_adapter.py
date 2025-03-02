
from models import Audio
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class EventAdapter:
    def extract_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement extract_data")

class RunPodEventAdapter(EventAdapter):
    def extract_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_data = event["input"]
            return {
                "user": input_data["user"],
                "filename": input_data["filename"],
                "conditioning": Audio(
                    prompt=input_data["prompt"],
                    seconds_start=input_data["seconds_start"],
                    seconds_total=input_data["seconds_total"]
                )
            }
        except KeyError as e:
            logger.error(f"Faltan datos en el evento: {e}")
            raise ValueError(f"Dato requerido no encontrado: {e}")

event_adapter = RunPodEventAdapter()