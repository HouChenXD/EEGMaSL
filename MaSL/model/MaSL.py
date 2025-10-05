import math
import torch
import torch.nn as nn
from mamba_ssm import  Mamba2 as Mamba 

class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes,dropout_prob=0):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class MaSLEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=1,
        n_fft=200,
        hop_length=100,
        dropout_prob = 0.2,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )

        self.mamba1 = Mamba(d_model=emb_size, d_state=128, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=emb_size, d_state=128, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=emb_size, d_state=128, d_conv=4, expand=2)
        self.mamba4 = Mamba(d_model=emb_size, d_state=128, d_conv=4, expand=2)
        self.positional_encoding = PositionalEncoding(emb_size)

        self.dropout_mmaba1 = nn.Dropout(p=dropout_prob)
        self.dropout_mmaba2 = nn.Dropout(p=dropout_prob)
        self.dropout_mmaba3 = nn.Dropout(p=dropout_prob)
        self.dropout_mmaba4 = nn.Dropout(p=dropout_prob)

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1), # [2, 2000]
            n_fft = self.n_fft, 
            hop_length = self.hop_length, 
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):  
            channel_spec_emb = self.stft(x[:, i : i + 1, :]) # input: [2, 1, 2000], output:[2, 101, 19]
            channel_emb = self.patch_embedding(channel_spec_emb) # [2, 19, 256]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1) # [2, 19 * 16 == 304, 256]
        
        emb = self.dropout_mmaba1(self.mamba1(emb))
        emb = self.dropout_mmaba2(self.mamba2(emb))
        emb = self.dropout_mmaba3(self.mamba3(emb))
        emb = self.dropout_mmaba4(self.mamba4(emb))
        
        emb = emb.mean(dim=1)
        return emb

# supervised classifier module
class MaSLClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()
        self.MaSL = MaSLEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.MaSL(x)
        x = self.classifier(x)
        return x

# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.MaSL = MaSLEncoder(emb_size, heads, depth, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.Dropout(kwargs['dropout'])
        )

    def forward(self, x, n_channel_offset=0):
        x_noaug,x_aug = x[0],x[1]
        
        # constrastive learning positive examples:emb,pred_emb
        emb = self.MaSL(x_aug, n_channel_offset) # perturb-positive examples
        emb = self.prediction(emb)  
        pred_emb = self.MaSL(x_noaug, n_channel_offset)
        pred_emb = self.prediction(pred_emb)
        return emb, pred_emb

# supervised pre-train module
class SupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4,n_channels=16,dropout_prob=0., **kwargs):
        super().__init__()
        self.MaSL = MaSLEncoder(emb_size=emb_size, heads=heads, depth=depth,n_channels=n_channels)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1,dropout_prob)
        self.classifier_crowd_source = ClassificationHead(emb_size, 1,dropout_prob)
        self.classifier_tuab = ClassificationHead(emb_size, 1,dropout_prob)
        self.classifier_tuev = ClassificationHead(emb_size, 6,dropout_prob)

    def forward(self, x, task="chb-mit"):
        x = self.MaSL(x)
        # print("AFTER ENCODER:\t", type(x))
        if task == "chb-mit":
            x = self.classifier_chb_mit(x)
            # print("IN TRAINING:\t",type(x))
        elif task == "crowd_source":
            x = self.classifier_crowd_source(x)
        elif task == "tuab":
            x = self.classifier_tuab(x)
        elif task == "tuev":
            x = self.classifier_tuev(x)
        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":
    x = torch.randn(2, 2, 2000)
    # model = MMB_MaSLEncoder()
    # out = model(x)
    # print(out.shape)

    model = UnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    out1, out2 = model(x)
    # print(out1.shape, out2.shape)
