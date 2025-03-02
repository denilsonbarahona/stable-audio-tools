from dataclasses import dataclass

@dataclass
class Audio:
    prompt: str
    seconds_start: int
    seconds_total: int