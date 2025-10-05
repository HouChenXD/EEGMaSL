import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample
from scipy.signal import butter, iirnotch, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter

RATE = 0.2

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y

class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y

class UnsupervisedPretrainLoader(torch.utils.data.Dataset):

    def __init__(self, root_shhs,root_tu):
        WINDOW_SIZE = 200
        
        self.SHHS_LENGTH = 6000
        self.TU_LENGTH = 2000
        # shhs dataset
        self.shhs_list = []
        self.root_shhs = root_shhs
        if root_shhs:
            # shhs_list
            self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        
        self.tu_list = []
        self.root_tu = root_tu
        if self.root_tu:
            process_datasets = ["TUEP","TUSL","TUAR","TUSZ"]
            for pr_dat in process_datasets:
                path_dir = os.path.join(self.root_tu,pr_dat,"processed")
                dat_list = [os.path.join(path_dir,pf) for pf in os.listdir(path_dir)]
                print(f"   NUMBER of {pr_dat}: {len(dat_list)}")
                self.tu_list.extend(dat_list)
        print("(tu) unlabeled data size:", len(self.tu_list))    

    def __len__(self):
        # return len(self.prest_list) + len(self.shhs_list)
        return len(self.shhs_list) + len(self.tu_list)
    
    def tu_load(self, index):
        sample_path = self.tu_list[index]
        sample = pickle.load(open(sample_path, "rb"))
        X = sample["X"]
        # X = torch.FloatTensor(X)
        return X,0
    
    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        return sample, 1 

    def __getitem__(self, index):
        if index < len(self.tu_list):
            return self.tu_load(index)
        else:
            index = index - len(self.tu_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    tu_samples, shhs_samples = [], []

    for sample, flag in batch:
        if flag == 0:
            tu_samples.append(sample)
        else:
            shhs_samples.append(sample)
    
    noaug_list_shhs,aug_list_shhs = [],[]
    noaug_list_tu,aug_list_tu = [],[]
    
    if len(shhs_samples) > 0:
        # frequency mixing to get positive samples
        samples = [torch.FloatTensor(sample) for sample in shhs_samples]
        samples = torch.stack(samples, 0) # batch
        shhs_samples_aug_np = freq_mix_augument(samples)
        shhs_samples_aug = resample_and_aug(shhs_samples_aug_np,target_length=6000)
        
        # origin sample
        shhs_samples_noaug = resample_and_aug(np.array(shhs_samples),target_length=6000)
        
        noaug_list_shhs.append(shhs_samples_noaug)
        aug_list_shhs.append(shhs_samples_aug)
        
    if len(tu_samples) > 0:
        # frequency mixing to get positive samples
        tu_samples = [torch.FloatTensor(sample) for sample in tu_samples]
        tu_samples = torch.stack(tu_samples, 0)
        tu_samples_aug_np = freq_mix_augument(tu_samples)
        tu_samples_aug = resample_and_aug(tu_samples_aug_np,target_length=2000)
        
        # origin sample
        tu_samples_noaug = resample_and_aug(np.array(tu_samples),target_length=2000)
        
        noaug_list_tu.append(tu_samples_noaug)
        aug_list_tu.append(tu_samples_aug)
    
    noaug_combined_tu = torch.cat(noaug_list_tu, dim=0)
    aug_combined_tu = torch.cat(aug_list_tu, dim=0)
    noaug_combined_shhs = torch.cat(noaug_list_shhs, dim=0)
    aug_combined_shhs = torch.cat(aug_list_shhs, dim=0)
    return noaug_combined_tu, aug_combined_tu,noaug_combined_shhs,aug_combined_shhs
    
    
    

def freq_mix_eeg_batch(batch, x2,rate, dim=1):
    
    x_f = torch.fft.fft(batch, dim=dim) 
    m = torch.rand(x_f.shape) < rate 

    amp = torch.abs(x_f)
    _,index = amp.sort(dim=dim,descending=True)
    dominant_mask = index > 3 
    m = torch.bitwise_and(m,dominant_mask)

    freal = x_f.real.masked_fill(m,0) 
    fimag = x_f.imag.masked_fill(m,0) 

    x2_f = torch.fft.fft(x2,dim=dim)

    m = torch.bitwise_not(m)
    freal2 = x2_f.real.masked_fill(m,0)  
    fimag2 = x2_f.imag.masked_fill(m,0)  

    freal += freal2
    fimag += fimag2

    x_f = torch.complex(freal, fimag)

    # TODO:remove abs
    x = torch.abs(torch.fft.ifft(x_f, dim=dim))

    return x

def freq_mix_augument(batch):
    """
    batch:(b,num_channels, sample_rate*duration)
        -shhs:(b,2,3750)
    """
    
    random_select_x2 = torch.roll(batch, shifts=1,dims=0)

    # TODO:rate=0.2,0.3,0.4,0.5
    batch_aug = freq_mix_eeg_batch(batch,random_select_x2,rate=RATE,dim=2)

    batch_aug_np = batch_aug.cpu().numpy()

     
    return batch_aug_np

def resample_and_aug(batch,target_length):
    # shhs: (2, 6000) resample to 200-move to pretrained code
    samples_aug = resample(batch, target_length, axis=-1)
    
    # normalize samples (remove the amplitude)-move to pretrained code
    samples_aug = samples_aug / (
        np.quantile(
            np.abs(samples_aug), q=0.95, method="linear", axis=-1, keepdims=True
        )
        + 1e-8
    )
    # generate samples and targets and mask_indices
    samples_aug = torch.FloatTensor(samples_aug)

    return samples_aug


class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, crowd_source_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        crowd_source_x,crowd_source_y = crowd_source_data
        self.crowd_source_size = len(crowd_source_x)
        self.crowd_source_x = crowd_source_x
        self.crowd_source_y = crowd_source_y

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)
        print("tuev_size:\t",self.tuev_size)
        print("chb_mit_size:\t",self.chb_mit_size)
        print("tuab_size:\t",self.tuab_size)
        print("crowd_source size:\t",self.crowd_source_size)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.crowd_source_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]  # (16, 2560)
        # print("X:\t",X.shape)
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1) # (16, 2000)
        
        
        # channel normalization
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        # print("X after norm:\t",X.shape) # (16, 2000)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def crowd_source_load(self, index):
        data = self.crowd_source_x[index]
        data = resample(data, 2000, axis=-1)
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = self.crowd_source_y[index]
        return samples, y, 2
    
    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.crowd_source_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.crowd_source_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.crowd_source_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.crowd_source_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    crowd_source_samples, crowd_source_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    # TODO: 
    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 2:
            crowd_source_samples.append(sample)
            crowd_source_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(crowd_source_samples) > 0:
        crowd_source_samples = torch.stack(crowd_source_samples)
        crowd_source_labels = torch.LongTensor(crowd_source_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (chb_mit_samples, chb_mit_labels),
        (crowd_source_samples, crowd_source_labels),
        (tuab_samples, tuab_labels),
    )

import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(logits, targets, alpha=0.95, gamma=2.0, reduction='mean'):
    """
    Binary Focal Loss implementation with logits.
    
    Args:
        logits (Tensor): Predicted logits (before sigmoid), shape (N, 1).
        targets (Tensor): Ground truth binary labels, shape (N, 1).
        alpha (float): Weighting factor for the class.
        gamma (float): Focusing parameter.
        reduction (str): Reduction method ('mean' or 'sum').

    Returns:
        Tensor: Computed loss.
    """
    targets = targets.unsqueeze(-1)
    targets = targets.type_as(logits)
    
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    p = torch.sigmoid(logits)
    
    p_t = p * targets + (1 - p) * (1 - targets)
    
    focal_weight = alpha * (1 - p_t) ** gamma
    
    loss = focal_weight * BCE_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()





