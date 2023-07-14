import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import random
import pickle
import albumentations
import pydicom
import copy
from transformers import get_linear_schedule_with_warmup
import timm
from timm.utils import AverageMeter
import glob
import sys
import time
import wandb
import warnings
import skimage
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage import io, exposure, data
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(".."))
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from functions.loss import weighted_loss
from functions.helper import save_model, gc_collect
from pretrainedmodels import se_resnext101_32x4d


def window(name, img, WL=950, WW=1900):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    try:
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
    except:
        X = (X*0).astype('uint8')
        print(name)
    return X

class PEDataset(Dataset):
    def __init__(self, df, bbox_dict, image_list, image_size, transform):
        self.df = df
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.image_size=image_size
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,index):

        study_id = self.df['StudyInstanceUID'][index]
        series_id = self.df['Slice'].values[index]
        
        data1 = pydicom.dcmread('../data/train_images/'+ str(study_id)+ f'/{int(series_id)}' + '.dcm')
        x1 = data1.pixel_array
        res = x1.shape[1]
        x1c = x1*data1.RescaleSlope+data1.RescaleIntercept

        x1 = np.expand_dims(window(study_id, x1c, WL=500, WW=1800), axis=2)
        x2 = np.expand_dims(window(study_id, x1c, WL=400, WW=650), axis=2)
        x3 = np.expand_dims(window(study_id, x1c, WL=80, WW=300), axis=2)
        
        th = threshold_otsu(x2) 
        x2 = x2>=th
        x2 = remove_small_objects(x2, 10)
        x = np.concatenate([x1, x2, x3], axis=2)

        bbox = self.bbox_dict.loc[self.bbox_dict['StudyInstanceUID'] == study_id]
        x = x[int(max(0, bbox['ymin'].values[0]*res*0.8)):int(min(res, bbox['ymax'].values[0]*res*1.2)),int(max(0, bbox['xmin'].values[0]*res*0.8)):int(min(res, bbox['xmax'].values[0]*res*1.2)),:]        
        
        x = cv2.resize(x, (self.image_size,self.image_size))
        x = self.transform(image=x)['image']
        x = x.transpose(2, 0, 1)
        
        if 'C1_fracture' in self.df:
            frac_targets = torch.as_tensor(self.df.iloc[index][['C1_fracture', 'C2_fracture', 'C3_fracture', 'C4_fracture',
                                                            'C5_fracture', 'C6_fracture', 'C7_fracture']].astype('float32').values)
            vert_targets = torch.as_tensor(self.df.iloc[index][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values)
            frac_targets = frac_targets * vert_targets  # we only enable targets that are visible on the current slice
        
        return x, frac_targets, vert_targets

class seresnext101(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 7)
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(2048, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(2048, 7),
        )
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        return self.nn_fracture(feature), self.nn_vertebrae(feature)
    
    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="set fold")
    parser.add_argument("--image-size", type=int, default=512, help="set image size")
    args = parser.parse_args()
    
    #local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device("cuda", 1)
    #torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.device = device

    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    df_train = pd.read_csv('../final_train_df2.csv')
    fold = args.fold
    #df_train = df.query('split != @fold').reset_index(drop=True)
    max_train_batches = len(df_train)
    
    bbox = pd.read_csv('../localizer/bbox_pred_512_all.csv')
    image_list_train = df_train['StudyInstanceUID']

    # hyperparameters
    learning_rate = 0.0004
    batch_size = 32
    image_size = args.image_size
    num_epoch = 1

    model = seresnext101()
    model.to(args.device)

    num_train_steps = int(len(image_list_train)/(batch_size*1)*num_epoch)   # 4 GPUs
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, epochs=1,
                                                    steps_per_epoch=min(num_train_steps, len(df_train)),
                                                    pct_start=0.3)
    #criterion = nn.BCEWithLogitsLoss().to(args.device)

    # training
    train_transform = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2, p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    # iterator for training
    scaler = torch.cuda.amp.GradScaler()
    datagen = PEDataset(df = df_train, bbox_dict=bbox, image_list=image_list_train, image_size=image_size, transform=train_transform)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen, shuffle = True, batch_size=batch_size, num_workers=4, pin_memory=False)
    frac_loss_weight = 2.
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j,(images, y_frac, y_vert) in tqdm(enumerate(generator), total=num_train_steps):
            if(j >= num_train_steps):
                break
            optimizer.zero_grad()  
            with torch.cuda.amp.autocast():
                y_frac_pred, y_vert_pred = model.forward(images.half().to(args.device))
                frac_loss = weighted_loss(y_frac_pred, y_frac.to(args.device))
                vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(args.device))
                loss = frac_loss_weight * frac_loss + vert_loss
                
                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    print(f'Bad loss, skipping the batch {j}')
                    del loss, frac_loss, vert_loss, y_frac_pred, y_vert_pred
                    gc_collect()
                    continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        save_model(f'seresnext101{ep}_{num_train_steps}full_otsu', model)


if __name__ == "__main__":
    main()
