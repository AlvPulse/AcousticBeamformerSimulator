import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TF warnings

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf

# Load YAMNet globally so it's not reloaded per call
_YAMNET_MODEL = None

def get_yamnet_model():
    global _YAMNET_MODEL
    if _YAMNET_MODEL is None:
        try:
            # We must set this for tfhub to work on some machines
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            import pkg_resources # Check if available, to avoid crashing earlier
            _YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        except Exception as e:
            print(f"Warning: Failed to load YAMNet: {e}")
            _YAMNET_MODEL = "failed"
    return _YAMNET_MODEL

def evaluate_lam_preservation(reference_audio, processed_audio, fs):
    """
    Evaluates whether the beamformed output maintains the same YAMNet
    embeddings and top-1 class prediction as the clean reference audio.
    """
    model = get_yamnet_model()
    if model == "failed" or model is None:
        return {"embedding_similarity": 0.0, "top1_match": False}

    # YAMNet requires 16kHz mono, values in [-1.0, +1.0]
    if fs != 16000:
        reference_audio = librosa.resample(reference_audio, orig_sr=fs, target_sr=16000)
        processed_audio = librosa.resample(processed_audio, orig_sr=fs, target_sr=16000)

    ref_norm = reference_audio / (np.max(np.abs(reference_audio)) + 1e-8)
    proc_norm = processed_audio / (np.max(np.abs(processed_audio)) + 1e-8)

    # Run Inference
    scores_ref, embeddings_ref, _ = model(ref_norm)
    scores_proc, embeddings_proc, _ = model(proc_norm)

    embeddings_ref = embeddings_ref.numpy()
    embeddings_proc = embeddings_proc.numpy()
    scores_ref = scores_ref.numpy()
    scores_proc = scores_proc.numpy()

    # Calculate Cosine Similarity of mean embeddings
    mean_emb_ref = np.mean(embeddings_ref, axis=0)
    mean_emb_proc = np.mean(embeddings_proc, axis=0)

    norm_ref = np.linalg.norm(mean_emb_ref)
    norm_proc = np.linalg.norm(mean_emb_proc)

    if norm_ref == 0 or norm_proc == 0:
        sim = 0.0
    else:
        sim = np.dot(mean_emb_ref, mean_emb_proc) / (norm_ref * norm_proc)

    # Top-1 class match
    top1_ref = np.argmax(np.mean(scores_ref, axis=0))
    top1_proc = np.argmax(np.mean(scores_proc, axis=0))

    return {
        "embedding_similarity": float(sim),
        "top1_match": bool(top1_ref == top1_proc)
    }
