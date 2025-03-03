import torch
import torchaudio
import librosa
from einops import rearrange
from huggingface_hub import login
from utils.config import config
from models import Audio
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools import get_pretrained_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login(config.get_hf_token())

model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    

def generate_audio(conditioning: Audio)-> torch.Tensor:
    conditioning_dict = [{
        "prompt": conditioning.prompt,
        "seconds_start": conditioning.seconds_start,
        "seconds_total": conditioning.seconds_total
    }]

    try:
        audio_tensor = generate_diffusion_cond(
            model,
            steps=350,
            cfg_scale=7,
            conditioning=conditioning_dict,
            batch_size=1,
            sample_size = conditioning.seconds_total * model_config["sample_rate"],
            seed=-1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            negative_conditioning=[{
                "prompt": "Low quality audio with poorly tuned instruments, atonal melodies, distorted sounds, excessive noise, crackling artifacts, unbalanced frequencies, muddy or muffled tones, lack of clarity, harsh dissonance, unnatural reverb, inconsistent rhythms, unwanted static, low fidelity, robotic or synthetic artifacts, overly compressed dynamics, and lack of stereo depth, unsuitable for professional jingles, tones, or sound effects",
                "seconds_start": conditioning.seconds_start,
                "seconds_total": conditioning.seconds_total
            }],
        )
        logger.info(f"end tensor")
        return audio_tensor
    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}")
        raise

def process_audio_to_wav(audio_tensor: torch.Tensor, temp_file: str = "temp_audio.wav") -> str:
    try:
        output = rearrange(audio_tensor, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(temp_file, output, model_config["sample_rate"])
        return temp_file
    except Exception as e:
        logger.error(f"Error processing audio to WAV: {str(e)}")
        raise

def process_audio_to_wav_librosa(audio_tensor: torch.Tensor, temp_file: str = "temp_audio.wav") -> str:
    try:
        output = rearrange(audio_tensor, "b d n -> d (b n)")
        output_np = output.cpu().numpy()
        output_normalized = librosa.util.normalize(output_np)

        if output_normalized.shape[0] == 1:
            output_normalized = output_normalized[0]
        elif output_normalized.shape[0] == 2:
            output_normalized = output_normalized.T
        else:
            raise ValueError(f"Expected 1 or 2 channels, got {output_normalized.shape[0]}")

        output_shifted = librosa.effects.pitch_shift(
            y=output_normalized,
            sr=model_config["sample_rate"],
            n_steps=config.get_nsteps()
        )
        if output_shifted.ndim == 1:
            output_shifted = output_shifted[None, :]
        else:
            output_shifted = output_shifted.T

        output_shifted = torch.tensor(output_shifted, dtype=torch.float32)
        output_shifted = output_shifted.clamp(-1, 1).mul(32767).to(torch.int16)

        torchaudio.save(temp_file, output_shifted, model_config["sample_rate"])
        logger.info(f"Audio guardado en {temp_file}")
        return temp_file
    except Exception as e:
        logger.error(f"Error processing audio to WAV: {str(e)}")
        raise