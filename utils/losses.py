import torch

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def batchMean_sisnrLoss(est, clean, eps=1e-8):
    batch_sisnr = sisnr(est, clean, eps)
    return -torch.mean(batch_sisnr)


def batchSum_MSE(y1, y2, _idx=2):
    # y1, y2: [N, C, F, T] or [N, F, T]
    loss = (y1-y2) ** _idx
    loss = torch.mean(torch.sum(loss, -2))
    return loss


def batchSum_relativeMSE(y1, y2, RL_epsilon=0.1, index_=2.0):
    # y1, y2: [N, C, F, T] ot [N, F, T]
    relative_loss = torch.abs(y1-y2) / (torch.abs(y1) + torch.abs(y2) + RL_epsilon)
    loss = torch.pow(relative_loss, index_)
    loss = torch.mean(torch.sum(loss, -2))
    return loss
