import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.runpod_api_key = os.getenv("RUNPOD_API_KEY")
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        self.gcp_client_email = os.getenv("GCP_CLIENT_EMAIL")
        self.nstep = os.getenv("N_STEPS")
        self.samplerTye = os.getenv('SAMPLER_TYPE')
        self.gcp_private_key = self.get_gcp_private_key()  # clave formateada correctamente

        if not self.hf_token:
            raise ValueError("HF_TOKEN no definido")
        if not self.runpod_api_key:
            raise ValueError("RUNPOD_API_KEY no definido")
        if not all([self.gcp_project_id, self.gcp_client_email, self.gcp_private_key]):
            logger.info("GCP credentials not fully set in env; falling back to ADC if available")

    def get_hf_token(self) -> str:
        return self.hf_token

    def get_runpod_api_key(self) -> str:
        return self.runpod_api_key

    def get_gcp_project_id(self) -> str:
        return self.gcp_project_id

    def get_gcp_client_email(self) -> str:
        return self.gcp_client_email

    def get_gcp_private_key(self) -> str:
        key = os.getenv("GCP_PRIVATE_KEY")
        return key.replace("\\n", "\n") if key else key
    
    def get_nsteps(self) -> int:
        return int(self.nstep)

    def getSamplerTye(self) -> str:
        return self.samplerTye

config = Config()
