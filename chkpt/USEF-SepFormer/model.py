import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.local.PositionalEncoding import PositionalEncoding


EPS = 1e-8


def select_norm(norm, dim, eps=1e-8):
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=eps)
    else:
        return nn.BatchNorm1d(dim)


class FiLM(nn.Module):
    def __init__(self, size=256):
        super(FiLM, self).__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x, aux):
        x = x * self.linear1(aux) + self.linear2(aux)
        return x


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Interblock(nn.Module):
    def __init__(
        self,
        d_model,
        intra_enc,
        inter_enc,
        max_length=20000,
    ):
        super(Interblock, self).__init__()

        self.intra_mdl = intra_enc
        self.inter_mdl = inter_enc

        self.intra_linear = nn.Linear(d_model, d_model)
        self.inter_linear = nn.Linear(d_model, d_model)

        self.intra_norm = select_norm("ln", d_model, 4)
        self.inter_norm = select_norm("ln", d_model, 4)

        self.pos_enc = PositionalEncoding(d_model, max_length)

    def forward(self, x):
        B, N, K, S = x.shape

        # intra_module
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra = self.intra_mdl(intra + self.pos_enc(intra))[0]
        intra = self.intra_linear(intra)
        intra = intra.view(B, S, K, N)
        intra = intra.permute(0, 3, 2, 1).contiguous()
        intra = self.intra_norm(intra) + x

        # inter_module
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter = self.inter_mdl(inter + self.pos_enc(inter))[0]
        inter = self.inter_linear(inter)
        inter = inter.view(B, K, S, N)
        inter = inter.permute(0, 3, 1, 2).contiguous()
        inter = self.inter_norm(inter)

        out = inter + intra

        return out


class Tar_Model(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        intra_enc,
        inter_enc,
        fusion_mdl,
        film,
        in_channels,
        out_channels,
        K,
        num_layers=2,
        norm="ln",
        num_spks=1,
        max_length=20000,
    ):
        super(Tar_Model, self).__init__()
        self.num_spks = num_spks
        self.num_layers = num_layers

        self.pos_enc = PositionalEncoding(out_channels, max_length)

        self.norm_m = select_norm(norm, in_channels, 3)
        self.conv1d1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.K = K
        self.encoder = encoder
        self.decoder = decoder

        self.conv2d = nn.Conv2d(out_channels, out_channels * num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

        self.fusion_mdl = fusion_mdl
        self.fusion_norm = select_norm("ln", out_channels, 3)
        self.film = film

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Interblock(
                        out_channels,
                        intra_enc,
                        inter_enc,
                    )
                )
            )

    def forward(self, input, aux):
        # [B, N, L]

        mix_w = self.encoder(input)
        aux = self.encoder(aux)

        x = self.norm_m(mix_w)
        aux = self.norm_m(aux)

        # [B, N, L]
        x = self.conv1d1(x)
        aux = self.conv1d1(aux)

        x = x.permute(0, 2, 1).contiguous()
        aux = aux.permute(0, 2, 1).contiguous()

        aux = self.fusion_mdl(x, aux)[0]
        x = self.film(x, aux)
        x = self.fusion_norm(x.permute(0, 2, 1).contiguous())

        x, gap_x = self._Segmentation(x, self.K)

        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)

        x = self.prelu(x)
        x = self.conv2d(x)
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)

        x = self._over_add(x, gap_x)
        x = self.output(x) * self.output_gate(x)
        x = self.end_conv1x1(x)
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        x = x.transpose(0, 1)

        mix_w = torch.stack([mix_w] * self.num_spks)
        x = mix_w * x

        est_source = torch.cat(
            [self.decoder(x[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )

        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source.squeeze(-1)

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input
