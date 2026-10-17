import os
import glob
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

from simulator.acoustics.array import load_array_geometry
from simulator.beamformers import LCMV_SMI # Wideband STFT LCMV

def process_chunked_multichannel(input_dir, output_file, array_geo_file, target_az=0.0, target_el=0.0, fs=16000, chunk_sec=2.0, overlap_sec=1.0):
    """
    Processes a directory of real multi-channel audio chunks (e.g., 2s files with 1s overlap)
    and stitches the beamformed output back into a continuous stream.

    Assumes files are named in sequential order (e.g., chunk_001.wav, chunk_002.wav, etc.)
    """
    if not os.path.exists(array_geo_file):
        raise FileNotFoundError(f"Array geometry file not found: {array_geo_file}")

    array_positions = load_array_geometry(array_geo_file)
    print(f"Loaded array geometry from {array_geo_file}: {array_positions.shape[0]} mics.")

    # Initialize the wideband LCMV beamformer
    bf = LCMV_SMI(fs=fs, n_fft=512, hop_length=256)

    # Find all chunks and sort them
    chunk_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not chunk_files:
        raise ValueError(f"No .wav files found in {input_dir}")

    print(f"Found {len(chunk_files)} chunk(s). Processing...")

    chunk_samples = int(chunk_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    hop_samples = chunk_samples - overlap_samples

    total_output_len = (len(chunk_files) - 1) * hop_samples + chunk_samples
    final_output = np.zeros(total_output_len)

    # Precompute a linear crossfade window for the overlap region
    # A standard Hanning window approach can also be used, but a simple linear crossfade
    # ensures amplitude sums to 1 in the overlap region.
    fade_in = np.linspace(0, 1, overlap_samples)
    fade_out = np.linspace(1, 0, overlap_samples)

    for i, file_path in enumerate(chunk_files):
        # Load multi-channel audio.
        # librosa loads as (Channels, Time) by default if mono=False
        audio, orig_sr = librosa.load(file_path, sr=fs, mono=False)

        # If it's single channel by mistake, we can't beamform
        if audio.ndim == 1:
            raise ValueError(f"File {file_path} is mono. Expected multi-channel.")

        # Ensure it matches the array geometry
        if audio.shape[0] != array_positions.shape[0]:
            print(f"Warning: Chunk {i} has {audio.shape[0]} channels, but array has {array_positions.shape[0]}.")

        print(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(file_path)}")

        # --- 1. BEAMFORMING ---
        # Pass the multi-channel matrix to the beamformer
        y_out = bf.process(audio, array_positions, target_az, target_el)

        # Ensure output length matches expected chunk size (pad if necessary)
        if len(y_out) < chunk_samples:
            y_out = np.pad(y_out, (0, chunk_samples - len(y_out)))
        elif len(y_out) > chunk_samples:
            y_out = y_out[:chunk_samples]

        # --- 2. OVERLAP-ADD (OLA) ---
        start_idx = i * hop_samples
        end_idx = start_idx + chunk_samples

        if i == 0:
            # First chunk: no fade in, fade out at the end
            y_out[-overlap_samples:] *= fade_out
            final_output[start_idx:end_idx] += y_out
        elif i == len(chunk_files) - 1:
            # Last chunk: fade in at the start, no fade out
            y_out[:overlap_samples] *= fade_in
            final_output[start_idx:end_idx] += y_out
        else:
            # Middle chunks: fade in at start, fade out at end
            y_out[:overlap_samples] *= fade_in
            y_out[-overlap_samples:] *= fade_out
            final_output[start_idx:end_idx] += y_out

    # Save final continuous output
    sf.write(output_file, final_output, fs)
    print(f"Finished! Stitched output saved to {output_file}")

if __name__ == "__main__":
    # Example Usage Block (Modify paths as needed for your real data)

    # 1. Path to the folder containing your 2-second overlapping .wav files
    INPUT_DIR = "data/real_multichannel_chunks"

    # 2. Path to output the final stitched 1-channel audio
    OUTPUT_FILE = "final_beamformed_output.wav"

    # 3. Path to your custom geometry file
    GEOMETRY_FILE = "array_geo.txt"

    # Ensure the input directory exists for the template to run without crashing initially
    os.makedirs(INPUT_DIR, exist_ok=True)

    # If the directory is empty, we'll create a couple dummy files just to prove it works
    if not glob.glob(os.path.join(INPUT_DIR, "*.wav")):
        print(f"No real data found in {INPUT_DIR}. Generating dummy multichannel files to test the script...")
        fs = 16000
        n_mics = 16 # Based on the array_geo.txt
        for i in range(3):
            dummy = np.random.randn(n_mics, fs * 2) * 0.1 # 2 seconds of noise
            sf.write(os.path.join(INPUT_DIR, f"chunk_{i:03d}.wav"), dummy.T, fs) # soundfile expects Time x Channels

    # Run the processor
    process_chunked_multichannel(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        array_geo_file=GEOMETRY_FILE,
        target_az=0.0,  # <-- Set your desired Look Azimuth here
        target_el=0.0,
        fs=16000,
        chunk_sec=2.0,
        overlap_sec=1.0
    )
