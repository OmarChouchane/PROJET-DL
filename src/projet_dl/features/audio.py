import numpy as np


def apply_cyclic_padding(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(audio) >= target_length:
        return audio[:target_length]
    repeats = target_length // len(audio)
    remainder = target_length % len(audio)
    out = np.tile(audio, repeats)
    if remainder:
        out = np.concatenate([out, audio[:remainder]])
    return out.astype(np.float32)


def extract_log_mel(audio: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
