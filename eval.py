import os
import sys

sys.path.append("../../..")
import argparse
import torch
import numpy as np
from tqdm import tqdm
from mir_eval.separation import bss_eval_sources
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq
from utils.average import AverageVal
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.data import Dataset
import librosa


class te_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, fs):
        self.mix = {x.split()[0]: x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]: x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]: x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref) == len(self.aux)

        wav_id = []
        for line in open(mix_scp):
            wav_id.append(line.split()[0])

        self.wav_id = wav_id
        self.fs = fs
        self.len = len(self.mix)

    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError

        utt = self.wav_id[index]
        mix_wav_path = self.mix[utt]
        target_wav_path = self.ref[utt]
        aux_wav_path = self.aux[utt]

        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs)
        target_wav, _ = librosa.load(target_wav_path, sr=self.fs)
        aux_wav, _ = librosa.load(aux_wav_path, sr=self.fs)

        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        return mix_wav, target_wav, aux_wav

    def __len__(self):
        return self.len


def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    # The replacement of "convolution_" is for compatibility with older SpeechBrain checkpoints.
    for k, v in model_info["model_state_dict"].items():
        name = k.replace("module.", "").replace(
            "convolution_", "convolution_module."
        )  # remove 'module.' prefix from DataParallel
        state_dict[name] = v
    model.load_state_dict(state_dict)

    return model


def main(config, args):
    model = config["modules"]["masknet"]
    model = load_pretrained_modules(model, args.chkpt_path)
    model.cuda()
    model.eval()

    testset = te_dataset(
        mix_scp=os.path.join(config[args.test_set], config["mix_scp"]),
        ref_scp=os.path.join(config[args.test_set], config["ref_scp"]),
        aux_scp=os.path.join(config[args.test_set], config["aux_scp"]),
        fs=config["sample_rate"],
    )
    test_dataloader = DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )

    fs = config["sample_rate"]

    # Helper function for PESQ - convert float to int16
    def to_int16(x):
        x = np.clip(x, -1, 1)
        return (x * 32767).astype(np.int16)

    # Select PESQ mode based on sampling rate
    pesq_mode = "wb" if fs == 16000 else "nb"

    with torch.inference_mode():
        total_cnt = 1
        sdrs_mix = AverageVal()
        sdrs_est = AverageVal()

        pesqs_mix = AverageVal()
        pesqs_est = AverageVal()

        sisnrs_mix = AverageVal()
        sisnrs_est = AverageVal()

        for i, (mix_wav, target_wav, emb_s1) in enumerate(test_dataloader):
            mix = mix_wav.cuda()
            embd = emb_s1.cuda()
            tar = target_wav.cuda()

            est_source = model(mix, embd)
            est_source = est_source.squeeze().cpu().numpy()

            mix = mix.squeeze().cpu().numpy()
            tar = tar.squeeze().cpu().numpy()

            end = min(est_source.size, mix.size, tar.size)

            mix_wav = mix[:end]
            tar_wav = tar[:end]
            est_wav = est_source[:end]

            sdr_mix = bss_eval_sources(tar_wav, mix_wav)[0][0]
            sdr_est = bss_eval_sources(tar_wav, est_wav)[0][0]
            sdrs_mix.update(sdr_mix)
            sdrs_est.update(sdr_est)

            sisnr_mix = si_snr(mix_wav, tar_wav)
            sisnr_est = si_snr(est_wav, tar_wav)
            sisnrs_mix.update(sisnr_mix)
            sisnrs_est.update(sisnr_est)

            # Convert to int16 for PESQ calculation
            ref16 = to_int16(tar_wav)
            mix16 = to_int16(mix_wav)
            est16 = to_int16(est_wav)

            # Updated PESQ calls with new parameter order (fs, ref, deg, mode)
            pesq_mix = pesq(fs, ref16, mix16, pesq_mode)
            pesq_est = pesq(fs, ref16, est16, pesq_mode)
            pesqs_mix.update(pesq_mix)
            pesqs_est.update(pesq_est)

            tqdm.write(
                "utt{}\t sdr-mix:({:.2f}){:.2f}\t sdr-est:({:.2f}){:.2f}\t sisnr-mix:({:.2f}){:.2f}\t sisnr-est:({:.2f}){:.2f}\t pesq-mix:({:.2f}){:.2f}\t pesq-est:({:.2f}){:.2f}\t ".format(
                    total_cnt,
                    sdrs_mix.val,
                    sdrs_mix.avg,
                    sdrs_est.val,
                    sdrs_est.avg,
                    sisnrs_mix.val,
                    sisnrs_mix.avg,
                    sisnrs_est.val,
                    sisnrs_est.avg,
                    pesqs_mix.val,
                    pesqs_mix.avg,
                    pesqs_est.val,
                    pesqs_est.avg,
                )
            )
            total_cnt += 1


def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm) ** 2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s) ** 2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Speech Separation")
    parser.add_argument("-c", "--config", type=str, default="", help="config file path")
    parser.add_argument(
        "-p", "--chkpt-path", type=str, default="", help="path to the chosen checkpoint"
    )
    parser.add_argument(
        "-t", "--test-set", type=str, default="", help="path to the test set"
    )
    args = parser.parse_args()

    for f in args.config, args.chkpt_path:
        assert os.path.isfile(f), "No such file: %s" % f

    with open(args.config, "r") as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    print("INFO: Loaded hparams from: {}".format(args.config))

    main(config, args)
