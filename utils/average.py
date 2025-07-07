import numpy as np


class AverageVal(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageVec(object):
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.avg = np.zeros(self.length)
        self.sum = np.zeros(self.length)
        self.count = np.zeros(self.length)

    def update(self, val, offset=0):
        ed = offset + len(val)
        assert ed <= self.length, "Out of Vec boundary."
        self.sum[offset:ed] += val
        self.count[offset:ed] += 1
        self.avg = self.sum / (self.count + 1e-10)


class AverageMat(object):
    def __init__(self, shape):
        self.shape = list(shape)
        self.reset()

    def reset(self):
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = np.zeros(self.shape)

    def update(self, val, offset=[0, 0]):
        x_st, y_st = offset
        x_ed = x_st + val.shape[0]
        y_ed = y_st + val.shape[1]
        assert x_ed <= self.shape[0] and y_ed <= self.shape[1], "Out of Mat boundary."
        self.sum[x_st:x_ed, y_st:y_ed] += val
        self.count[x_st:x_ed, y_st:y_ed] += 1
        self.avg = self.sum / (self.count + 1e-10)
