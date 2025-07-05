import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import librosa

class tr_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, dur, fs):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]:x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]:x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref) == len(self.aux)
        
        wav_id = []
        for l in open(mix_scp):
            wav_id.append(l.split()[0])
         
        self.wav_id = wav_id
        self.tlen = dur * fs
        
        self.fs = fs
        self.len = len(self.mix)
    
    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        else:
            y = y[offset:offset+tlen]
        return y 
    
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
        exclude = [utt.split('_')[0]+'.wav', utt.split('_')[2]+'.wav']
        aux_list = os.listdir(os.path.dirname(self.aux[utt]))
        aux_wav_path = os.path.join(os.path.dirname(self.aux[utt]), random.choice([x for x in aux_list if x not in exclude]))
        
        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs)
        target_wav, _ = librosa.load(target_wav_path, sr=self.fs)
        aux_wav, _ = librosa.load(aux_wav_path, sr=self.fs)

        offset = random.randint(0, max(len(target_wav) - self.tlen, 0))
        target_wav = self._trun_wav(target_wav, self.tlen, offset)
        mix_wav = self._trun_wav(mix_wav, self.tlen, offset)

        offset = random.randint(0, max(len(aux_wav) - self.tlen + 8000, 0))
        aux_wav = self._trun_wav(aux_wav, self.tlen - 8000, offset) # aux_len = 3s
        
        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        source_len = np.array([target_wav.shape[-1]])
        source_len = torch.from_numpy(source_len)

        return mix_wav, target_wav, aux_wav, source_len
    
    def __len__(self):
        return self.len


class te_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, fs):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]:x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]:x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref) == len(self.aux)
        
        wav_id = []
        for l in open(mix_scp):
            wav_id.append(l.split()[0])
         
        self.wav_id = wav_id
        self.fs = fs
        self.len = len(self.mix)

    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        else:
            y = y[offset:offset+tlen]
        return y 
    
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

        source_len = np.array([target_wav.shape[-1]])
        source_len = torch.from_numpy(source_len)

        return mix_wav, target_wav, aux_wav, source_len
    
    def __len__(self):
        return self.len