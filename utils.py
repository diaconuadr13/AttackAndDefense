# utils.py
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
import soundfile as sf

# Global constant for fixed audio length (1 second @ 16kHz)
FIXED_LENGTH = 16000 

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, data_dir: str = "./data"):

        os.makedirs(data_dir, exist_ok=True)
        super().__init__(data_dir, download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n: int):
        filepath = self._walker[n]
        data, sample_rate = sf.read(filepath)
        waveform = torch.from_numpy(data).float()
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        
        relpath = os.path.relpath(filepath, self._path)
        label, filename = os.path.split(relpath)
        
        return waveform, sample_rate, label, "speaker_id", 0

def get_labels(dataset):
    all_labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    return sorted(all_labels)

def label_to_index(word, labels):
    return torch.tensor(labels.index(word))

def collate_fn(batch):
    tensors, targets = [], []
    transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=32)
    labels_list = get_labels(None) # use fixed list

    for waveform, _, label, *_ in batch:
        # 1. Force Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 2. Force Fixed Length (Pad or Truncate to 16000)
        if waveform.shape[1] < FIXED_LENGTH:
            # Pad
            padding = FIXED_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > FIXED_LENGTH:
            # Truncate
            waveform = waveform[:, :FIXED_LENGTH]

        # 3. Transform to MFCC
        tensors += [transform(waveform).squeeze(0).transpose(0, 1)]
        targets += [label]

    # Stack tensors directly (they are now all the same size)
    tensors = torch.stack(tensors)
    
    # Add channel dimension (B, 1, Time, Freq)
    tensors = tensors.unsqueeze(1) 
    
    targets = torch.stack([label_to_index(t, labels_list) for t in targets])

    return tensors, targets