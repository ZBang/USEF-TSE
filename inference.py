"""
USEF-TSE Inference Script

This script provides a command-line interface for performing target speaker extraction
using pre-trained USEF-TSE models (SepFormer or TFGridNet).

It takes a mixed audio file (containing the target speaker and other sounds) and a
reference audio file (a clean sample of the target speaker's voice) as input.
The script then separates the target speaker's voice from the mixture and saves
the resulting audio to a specified output file.

The script can automatically download the required model checkpoints from HuggingFace
if they are not found locally. It supports inference on CPU, CUDA, and MPS (Apple Silicon)
devices, with optional FP16 precision for performance improvement on GPUs. For very
long audio files, a segmentation option is available to process the audio in chunks,
reducing memory consumption.

CLI Usage:
----------
For help:
  python inference.py --help

Basic usage with SepFormer model:
  python inference.py --type sepformer \\
                      --mix /path/to/your/mix.wav \\
                      --ref /path/to/your/ref.wav \\
                      --out /path/to/output/extracted.wav

Basic usage with TFGridNet model:
  python inference.py --type tfgridnet \\
                      --mix /path/to/your/mix.wav \\
                      --ref /path/to/your/ref.wav \\
                      --out /path/to/output/extracted.wav

Using segmentation for long audio files (e.g., 10-second chunks) and FP16:
  python inference.py --type sepformer --mix long_mix.wav --ref ref.wav --out out.wav --segment-sec 10 --fp16

"""

import os
import time
import logging
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from typing_extensions import Annotated

import typer
import torch
import librosa
import numpy as np
import soundfile as sf
from hyperpyyaml import load_hyperpyyaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent


def download_chkpt_if_needed(chkpt_path: str, fetch_url: str):
    """Checks if a checkpoint exists, and if not, downloads it from HuggingFace."""
    if os.path.exists(chkpt_path):
        logger.info(f"Checkpoint found: {chkpt_path}")
        return

    logger.warning(f"Checkpoint not found at {chkpt_path}. Attempting to download...")

    # Ensure the target directory exists
    chkpt_dir = os.path.dirname(chkpt_path)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
        logger.info(f"Created directory: {chkpt_dir}")

    # Construct the download URL
    logger.info(f"Downloading from: {fetch_url}")
    try:
        # Using torch.hub to download with progress bar
        torch.hub.download_url_to_file(fetch_url, chkpt_path, progress=True)
        logger.info("Download successful.")
    except Exception as e:
        logger.error(f"Fatal: Failed to download checkpoint from {fetch_url}")
        logger.error(f"Error: {e}")
        # Exit if download fails, as the script cannot proceed.
        raise typer.Exit(code=1)


def load_pretrained_modules(model, ckpt_path):
    """
    Loads pretrained modules from a checkpoint file.
    """
    logger.info(f"Loading pretrained modules from {ckpt_path}")
    model_info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = OrderedDict()
    for k, v in model_info["model_state_dict"].items():
        name = k.replace("module.", "").replace("convolution_", "convolution_module.")
        state_dict[name] = v
    model.load_state_dict(state_dict)
    logger.info(f"Successfully loaded checkpoint from: {ckpt_path}")
    return model


class ModelType(str, Enum):
    sepformer = "sepformer"
    tfgridnet = "tfgridnet"


def main(
    model_type: Annotated[
        ModelType,
        typer.Option(
            "--type",
            help="Model type for inference: 'sepformer' or 'tfgridnet'.",
            rich_help_panel="Required Parameters",
            case_sensitive=False,
        ),
    ],
    mix_audio_path: Annotated[
        str,
        typer.Option(
            "--mix",
            help="Path to the mixed audio file.",
            rich_help_panel="Required Parameters",
        ),
    ],
    ref_audio_path: Annotated[
        str,
        typer.Option(
            "--ref",
            help="Path to the reference speaker audio file.",
            rich_help_panel="Required Parameters",
        ),
    ],
    output_path: Annotated[
        str,
        typer.Option(
            "--out",
            help="Path to save the output extracted audio.",
            rich_help_panel="Required Parameters",
        ),
    ],
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force use of CPU for inference.",
            rich_help_panel="Optional Parameters",
        ),
    ] = False,
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16",
            help="Use half-precision (FP16) for inference on supported GPUs (CUDA/MPS).",
            rich_help_panel="Optional Parameters",
        ),
    ] = False,
    segment_seconds: Annotated[
        float,
        typer.Option(
            "--segment-sec",
            help="Process audio in segments of this duration (in seconds) to save memory. 0 means process whole file.",
            rich_help_panel="Optional Parameters",
        ),
    ] = 0.0,
):
    """
    USEF-TSE Inference Script for Target Speaker Extraction.
    Performs target speaker extraction on a given mixed audio file using a reference audio.
    """
    # Start timer
    start_time = time.time()

    logger.info("Starting USEF-TSE inference")
    logger.info(
        f"Parameters - model_type: {model_type}, mix: {mix_audio_path}, ref: {ref_audio_path}, out: {output_path}"
    )
    logger.info(
        f"Options - cpu: {cpu}, fp16: {fp16}, segment_seconds: {segment_seconds}"
    )

    # 1. Determine config and checkpoint paths based on model type

    if model_type == ModelType.tfgridnet:
        config_path = REPO_DIR / "chkpt" / "USEF-TFGridNet" / "config.yaml"
        chkpt_path = (
            REPO_DIR / "chkpt" / "USEF-TFGridNet" / "wsj0-2mix" / "temp_best.pth.tar"
        )
        fetch_url = "https://huggingface.co/ZBang/USEF-TSE/resolve/main/chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar"
    elif model_type == ModelType.sepformer:
        config_path = REPO_DIR / "chkpt" / "USEF-SepFormer" / "config.yaml"
        chkpt_path = (
            REPO_DIR / "chkpt" / "USEF-SepFormer" / "wsj0-2mix" / "temp_best.pth.tar"
        )
        fetch_url = "https://huggingface.co/ZBang/USEF-TSE/resolve/main/chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar"

    config_path = str(config_path)
    chkpt_path = str(chkpt_path)

    logger.info(f"Selected model type: {model_type.value}")
    logger.info(f"Using config: {config_path}")
    logger.info(f"Using checkpoint: {chkpt_path}")

    # 2. Download checkpoint if it doesn't exist locally
    download_chkpt_if_needed(chkpt_path, fetch_url)

    use_gpu = not cpu

    # 3. Load configuration and model
    logger.info("Loading model and configuration...")
    with open(config_path, "r") as f:
        config = load_hyperpyyaml(f.read())
    model = config["modules"]["masknet"]

    # 4. Load pretrained weights
    model = load_pretrained_modules(model, chkpt_path)

    # 5. Set device (GPU/CPU) and evaluation mode
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU (CUDA) for inference. Device: {gpu_name}")
        # Reset memory stats before inference
        torch.cuda.reset_peak_memory_stats(device)
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
        logger.info("Using GPU (MPS) for inference.")
    else:
        device = torch.device("cpu")
        model.to(device)
        logger.info("Using CPU for inference.")
    model.eval()

    # Determine autocast settings
    # Only enable for CUDA/MPS and if fp16 flag is true
    use_autocast = fp16 and device.type in ["cuda", "mps"]
    dtype = torch.float16 if use_autocast else torch.float32
    if use_autocast:
        logger.info(f"FP16/autocast enabled for inference on {device.type.upper()}.")

    # 6. Load and preprocess audio
    logger.info("Loading and preprocessing audio...")
    sample_rate = config["sample_rate"]
    mix_wav, _ = librosa.load(mix_audio_path, sr=sample_rate)
    ref_wav, _ = librosa.load(
        ref_audio_path, sr=sample_rate
    )  # ref audio is usually short, load fully
    ref_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)  # [1, L_ref]

    # 7. Perform inference
    logger.info("Performing inference...")
    if segment_seconds > 0:
        logger.info(f"Processing in {segment_seconds}s segments to conserve memory.")
        segment_samples = int(segment_seconds * sample_rate)
        overlap_samples = segment_samples // 10  # 10% overlap
        step = segment_samples - overlap_samples

        est_wav_full = np.array([])

        with torch.inference_mode():  # More efficient than no_grad()
            for start in range(0, len(mix_wav), step):
                end = start + segment_samples
                mix_chunk = mix_wav[start:end]

                if (
                    len(mix_chunk) < sample_rate * 0.5
                ):  # Skip very short trailing segments
                    continue

                logger.debug(
                    f"Processing segment from {start / sample_rate:.2f}s to {end / sample_rate:.2f}s"
                )
                mix_tensor = torch.from_numpy(mix_chunk).unsqueeze(0).to(device)

                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=use_autocast
                ):
                    est_source_chunk = model(mix_tensor, ref_tensor)
                est_wav_chunk = est_source_chunk.squeeze(0).cpu().numpy()

                # Overlap-add windowing to stitch results smoothly
                if start == 0:
                    est_wav_full = est_wav_chunk
                else:
                    # Note: A more sophisticated overlap-add with a window function (e.g., hann) might give better results,
                    # but linear cross-fade is a good starting point.
                    # Cross-fade in the overlap region
                    fade_len = min(
                        overlap_samples, len(est_wav_full) - step, len(est_wav_chunk)
                    )
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    # Apply fade to the end of the existing signal and start of the new chunk
                    est_wav_full[-fade_len:] *= fade_out
                    est_wav_chunk[:fade_len] *= fade_in

                    # Add the overlapping part and append the rest
                    est_wav_full[-fade_len:] += est_wav_chunk[:fade_len]
                    est_wav_full = np.concatenate(
                        (est_wav_full, est_wav_chunk[fade_len:])
                    )
        est_wav = est_wav_full
    else:
        with torch.inference_mode():
            mix_tensor = torch.from_numpy(mix_wav).unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=use_autocast
            ):
                est_source = model(mix_tensor, ref_tensor)
            est_wav = est_source.squeeze(0).cpu().numpy()

    # 8. Post-process and save result
    logger.info(f"Inference complete. Saving extracted audio to {output_path}")
    sf.write(output_path, est_wav, sample_rate)

    # --- Add performance stats ---
    end_time = time.time()
    total_time = end_time - start_time

    logger.info("--- Performance Stats ---")
    logger.info(f"Total execution time: {total_time:.2f} seconds")

    if device.type == "cuda":
        # torch.cuda.max_memory_allocated returns bytes, convert to GB
        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        logger.info(f"Peak GPU memory usage: {max_mem_gb:.2f} GB")
    logger.info("-------------------------")


if __name__ == "__main__":
    typer.run(main)
#
