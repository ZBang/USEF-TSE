# modal_entry.py
import logging
import pathlib
import subprocess

import modal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

APP_NAME = "usef-tse-infer"
REPO_ROOT = pathlib.Path(__file__).parent.resolve()

# 1) Build an image with your project code + deps
image = (
    modal.Image.debian_slim()  # lightweight base
    .workdir("/root/USEF-TSE")  # set working directory
    .pip_install_from_pyproject(str(REPO_ROOT / "pyproject.toml"))  # project deps
    .add_local_dir(
        local_path=str(REPO_ROOT),  # copy entire repo
        remote_path="/root/USEF-TSE",  # where it appears inside the container
        ignore=["*.venv"],
        # keep_git=True  # optional: include .git
    )
)

app = modal.App(APP_NAME)
vol_output = modal.Volume.from_name("usef_tse_output", create_if_missing=True)


# 2) Wrap the actual Typer CLI in a Modal function
@app.function(
    image=image,
    timeout=60 * 60,
    cpu=2.0,
    gpu="L40S",  # specify GPU type if needed
    volumes={
        "/output": vol_output,
    },
)
def run_inference(model_type: str, mix: str, ref: str, out: str = "/output/output.wav"):
    """
    Thin wrapper that simply calls the existing Typer entrypoint
    inside the container so we don't rewrite inference_modal.py.
    """

    logger.info(
        f"Starting inference with model_type={model_type}, mix={mix}, ref={ref}, out={out}"
    )

    cmd = [
        "python",
        "inference.py",
        "--type",
        model_type,
        "--mix",
        mix,
        "--ref",
        ref,
        "--out",
        out,  # output path inside the container
    ]

    vol_output.commit()

    try:
        subprocess.check_call(cmd)
        logger.info("Inference subprocess completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference subprocess failed with return code {e.returncode}")
        raise


# 3) Local CLI → remote execution
@app.local_entrypoint()
def main(
    type: str = "tfgridnet",
    mix: str = "./mix.wav",
    ref: str = "./enrollment.wav",
    out: str = "/output/output.wav",
):
    """
    Called by:  modal run modal_entry.py --type tfgridnet ...
    Works exactly like the original script, but on Modal's infra.
    """
    logger.info(
        f"Starting remote inference with parameters: type={type}, mix={mix}, ref={ref}, out={out}"
    )

    try:
        run_inference.remote(type, mix, ref, out)
        logger.info(f"✅ Finished! Extracted vocals written to {out}")
    except Exception as e:
        logger.error(f"Remote inference failed: {e}")
        raise
