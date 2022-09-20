import numpy as np
import pandas as pd
import altair as alt

import shutil
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

dev = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


# ==============
# DATA
# ==============

data_path = Path().resolve().parent / 'data'

# tags

img_path, top_path = [data_path /
 f'{f}_tags.csv'for f in ['img', 'top']]

all_labels = pd.read_csv(
    img_path, converters={'tags': eval}
).sort_values('id').reset_index(drop=True)

label_converter = pd.read_csv(top_path).squeeze()


def lbls2proba(labels):
    return torch.FloatTensor(label_converter.apply(
            lambda name: .9 if name in labels else .1))

# images


preprocess = T.Compose([
    T.Resize(384),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# ds class


class DanbooruDataset(Dataset):

    def __init__(self, label_data, img_dir,
                 transform=preprocess,
                 target_transform=lbls2proba):

        self.label_data = label_data
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        img_id = self.label_data.iloc[idx, 0]
        img_path = Path(self.img_dir) / f'{img_id}.jpg'
        image = self.transform(Image.open(img_path).convert('RGB'))
        labels = self.target_transform(self.label_data.iloc[idx, 1])
        return image.to(dev), labels.to(dev)


def get_random_sample(ds, size, rng=np.random.default_rng()):
    return Subset(ds, rng.integers(len(ds), size=size))


# ==============
# MODEL
# ==============

torch.hub.set_dir('C:/Dev/.cache/pytorch')

# additional layers on top of base nn


def bn_drop_lin(in_size, out_size):
    return nn.Sequential(
        nn.BatchNorm1d(
            in_size,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_size, out_size))

# model class

class Tagger(nn.Module):
    def __init__(self):
        super(Tagger, self).__init__()

        self.out_classes = len(label_converter)

        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        layers = list(backbone.children())
        in_features = layers[-1][-1].in_features
        self.base = nn.Sequential(*layers[:-1]).eval()

        self.classifier = bn_drop_lin(in_features, self.out_classes)

    def forward(self, t_in):

        t_base = F.leaky_relu(self.base(t_in))[:, :, 0, 0]
        
        t = F.leaky_relu(self.classifier(t_base))
        
        t_rs = t.reshape([len(t_base), self.out_classes])
        t_cl = torch.clamp(t_rs, -10, 10)

        t_out = torch.sigmoid(t_cl)

        return t_out