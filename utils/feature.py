import torch
import torch.nn as nn
import random
from torchaudio import transforms


class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, x, is_aug=[]):
        out = self.fbankCal(x)[..., :-1]  # 舍弃最后一个0.01s，对齐长度
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)

        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1] / 8, out.shape[1] / 4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start + offset] = (
                    out[i][start : start + offset] * random.random() / 2
                )

        return out

    def forward_sample(self, x, is_aug=False):
        x = self.forward(x.unsqueeze(0), [is_aug])
        x = x.squeeze(0)

        return x


class STFT(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y):
        num_dims = y.dim()
        assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)  # [B * C ,T]

        complex_stft = torch.stft(
            y,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )
        _, num_freqs, num_frames = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

        # print(complex_stft)

        mag = torch.abs(complex_stft)
        phase = torch.angle(complex_stft)
        real = complex_stft.real
        imag = complex_stft.imag
        return mag, phase, real, imag, complex_stft


class iSTFT(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256, length=None):
        super(iSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length

    def forward(self, features, input_type):
        if input_type == "real_imag":
            # the feature is (real, imag) or [real, imag]
            assert isinstance(features, tuple) or isinstance(features, list)
            real, imag = features
            features = torch.complex(real, imag)
        elif input_type == "complex":
            assert torch.is_complex(features), "The input feature is not complex."
        elif input_type == "mag_phase":
            # the feature is (mag, phase) or [mag, phase]
            assert isinstance(features, tuple) or isinstance(features, list)
            mag, phase = features
            features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        else:
            raise NotImplementedError(
                "Only 'real_imag', 'complex', and 'mag_phase' are supported."
            )

        return torch.istft(
            features,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=features.device),
            length=self.length,
        )
