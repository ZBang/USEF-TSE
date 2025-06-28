# USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction

[![Paper](https://img.shields.io/badge/Paper-red?&logo=arxiv)](https://arxiv.org/pdf/2409.02615)

Official Implementation of USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction.

Currently, this repository only provides inference code. The complete training code will be open-sourced after the paper revisions are finalized.

To refer to the model class, check [models](./models/) directly.

## Inference
To run inference on the test sets mentioned in the paper, for example, you can run

```shell
bash eval.sh
```
Please note that before performing model inference, you need to check the data paths (refer to the files in the [data](./data/) folder for details).

## Model Checkpoint

Our USEF-TSE checkpoints can be downloaded [here](https://huggingface.co/ZBang/USEF-TSE).

## LICENSE

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

