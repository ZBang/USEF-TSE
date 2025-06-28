import torch
import torch.nn as nn
import copy

from models.local.TFgridnet import GridNetV2Block

EPS = 1e-8


class Tar_Model(nn.Module):
    def __init__(
        self,
        stft,
        istft,
        real_att,
        n_freqs,
        hidden_channels,
        n_head,
        emb_dim,
        emb_ks,
        emb_hs,
        num_layers=6,
        eps=1e-5,
    ):
        super(Tar_Model, self).__init__()
        self.num_layers = num_layers

        self.stft = stft
        self.istft = istft

        self.att = real_att

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.deconv = nn.ConvTranspose2d(2 * emb_dim, 2, ks, padding=padding)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    GridNetV2Block(
                        2 * emb_dim,
                        emb_ks,
                        emb_hs,
                        n_freqs,
                        hidden_channels,
                        n_head,
                        approx_qk_dim=512,
                        activation="prelu",
                    )
                )
            )

    def forward(self, input, aux):
        # [B, N, L]
        input = input.unsqueeze(1)
        aux = aux.unsqueeze(1)

        std = input.std(dim=(1, 2), keepdim=True)
        input = input / std

        mix_c = self.stft(input)[-1]
        aux_c = self.stft(aux / aux.std(dim=(1, 2), keepdim=True))[-1]

        mix_ri = torch.cat([mix_c.real, mix_c.imag], dim=1)
        mix_ri = mix_ri.permute(0, 1, 3, 2).contiguous()

        aux_ri = torch.cat([aux_c.real, aux_c.imag], dim=1)
        aux_ri = aux_ri.permute(0, 1, 3, 2).contiguous()

        mix_ri = self.conv(mix_ri)
        aux_ri = self.conv(aux_ri)

        aux_ri = self.att(mix_ri, aux_ri)

        x = torch.cat([mix_ri, aux_ri], dim=1)

        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)

        x = self.deconv(x)

        out_r = x[:, 0, :, :].permute(0, 2, 1).contiguous()
        out_i = x[:, 1, :, :].permute(0, 2, 1).contiguous()

        est_source = self.istft((out_r, out_i), input_type="real_imag").unsqueeze(1)

        est_source = est_source * std

        return est_source.squeeze(1)
