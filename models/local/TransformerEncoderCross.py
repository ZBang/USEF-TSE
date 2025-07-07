import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from models.local import attention
from models.local import normalization


class TransformerEncoderLayerCross(nn.Module):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=False,
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )

        self.pos_ffn = attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        # self.MLP1 = torch.nn.Linear(d_model, d_model)
        # self.MLP2 = torch.nn.Linear(d_model, d_model)
        # self.MLP3 = torch.nn.Linear(d_model, d_model)

        # self.conv1d1 = torch.nn.Conv1d(2*d_model, d_model, 1)
        # self.conv1d2 = torch.nn.Conv1d(2*d_model, d_model, 1)
        # self.conv1d3 = torch.nn.Conv1d(2*d_model, d_model, 1)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        embd,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        # q = self.conv1d1(torch.cat([src1,embd], -1).permute(0,2,1).contiguous())
        # k = self.conv1d2(torch.cat([src1,embd], -1).permute(0,2,1).contiguous())
        # v = self.conv1d3(torch.cat([src1,embd], -1).permute(0,2,1).contiguous())

        output, self_attn = self.self_att(
            # query = q.permute(0,2,1).contiguous(),
            # key = k.permute(0,2,1).contiguous(),
            # value = v.permute(0,2,1).contiguous(),
            # query = self.MLP1(src1 + embd)
            # key = self.MLP2(src1 + embd)
            # value = self.MLP3(src1 + embd)
            src1,
            embd,
            embd,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class TransformerEncoderCross(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=0.0,
        attention_type="regularMHA",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayerCross(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src,
        embd,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    embd,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst
