# USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction

[![Paper](https://img.shields.io/badge/Paper-red?&logo=arxiv)](https://arxiv.org/pdf/2409.02615)

Official Implementation of USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction.

Currently, this repository only provides inference code. The complete training code will be open-sourced after the paper revisions are finalized.

To refer to the model class, check [models](./models/) directly.

## Inference

### Installation
1. Clone this repository
```sh
git clone https://github.com/ZBang/USEF-TSE.git
```
2. Install python dependencies
You can use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the dependencies. In addition, you may also use other tools such as pip, conda, venv, or any other tools you like.

If you use uv, replace all `python <code.py>` with `uv run <code.py>` and run this to install the dependencies.
```sh
uv sync
```

By the way, you can use uv inside a conda environment.

If you don't use uv or poetry or any other modern python dependency management tool, run the following to install the dependencies.
```sh
pip install -r requirements.txt --use-pep517
```


### Inference on the test set
To run inference on the test sets mentioned in the paper, for example, you can run the following command:

If you use uv:
```sh
export CUDA_VISIBLE_DEVICES=2

uv run eval.py  --config chkpt/USEF-TFGridNet/config.yaml \
    --chkpt-path chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar \
    --test-set wsj0-2mix #['wsj0-2mix', 'wham!', 'whamr!']
```
If you don't use uv:
```shell
bash eval.sh
```

Please note that before performing model inference, you need to check the data paths (refer to the files in the [data](./data/) folder for details).

### Inference on your own data

Replace uv with python if you use something other than uv.
```sh
uv run inference.py \
    --type tfgridnet \
    --mix ./mix_audio.wav \
    --ref ./enrollment.wav \
    --out ./output.wav
```
Use `uv run inference.py --help` for help.
- `--type` can either be `sepformer` or `tfgridnet`: the pretrained model provided [here](https://huggingface.co/ZBang/USEF-TSE/tree/main). `sepformer` is `chkpt/USEF-SepFormer/wsj0-2mix` and `tfgridnet` is `chkpt/USEF-TFGridNet/wsj0-2mix`. The model will be automatically downloaded from huggingface if not locally available.
- `--mix`: the path to the mix audio.
- `--ref`: the path to the enrollment audio.
- `--out`: the path to save the output

### Inference on your own data on Modal.com
Modal.com provides some free GPU inference every months.

Use this to login.
```sh
uv run modal setup
```

and run
```sh
uv run modal run modal_entry.py \
    --type tfgridnet \
    --mix ./mix.wav \
    --ref ./ref.wav \
    --out /output/output_tfgridnet.wav
```
- mix audio and reference (enrollment) audio need to be within the project directory
- out directory is in the modal volume. Don't change it unless you know what you are doing.



## Model Checkpoint

Our USEF-TSE checkpoints can be downloaded [here](https://huggingface.co/ZBang/USEF-TSE).

