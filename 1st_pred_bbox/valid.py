import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from efficientnet_pytorch import EfficientNet
import random
from sklearn.metrics import roc_auc_score
import pickle
import pydicom
from timm.utils import AverageMeter
import pylibjpeg

def window(x, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x


class PEDataset(Dataset):
    def __init__(self, df, bbox_dict, image_list, target_size):
   
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
        self.df = df
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,index):
        data = pydicom.dcmread(self.image_list[index])
        x = data.pixel_array.astype(np.float32)
        res = x.shape[1]
        x = x*data.RescaleSlope+data.RescaleIntercept
        x1 = window(x, WL=950, WW=1900)
        x = np.zeros((x1.shape[0], x1.shape[1], 3), dtype=np.float32)
        x[:,:,0] = x1
        x[:,:,1] = x1
        x[:,:,2] = x1
        x = cv2.resize(x, (self.target_size,self.target_size))
        bboxes = self.bbox_dict[str(self.df['StudyInstanceUID'].values[index]) + '/' + str(self.df['slice_number'].values[index])]
        bboxes_norm = [[max(0.0, bboxes[0] /res), max(0.0, bboxes[1]/res), min(1.0, bboxes[2]/res),min(1.0, bboxes[3]/res)]]
        x = x.transpose(2, 0, 1)
        y = torch.from_numpy(np.array(bboxes_norm)).squeeze()
        return x, y
    
    

class efficientnet(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.net._fc.in_features
        self.last_linear = nn.Linear(in_features, 4)
    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--split-ind', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--finetune-from', type=str, default='')
    parser.add_argument('--multi-class', action='store_true', default=False)
    parser.add_argument('--no-patch', action='store_true', default=False)
    parser.add_argument('--loss', default='', type=str)

    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-patches', type=int, default=1)
    parser.add_argument('--grad-accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--min-lr', type=float, default=1e-8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--weight-decay-start', type=float, default=0.001)
    parser.add_argument('--weight-decay-end', type=float, default=0.1)

    parser.add_argument('--pre-size', type=int, default=0)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--stride-size', type=int, default=-1)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--train-augmentation', type=str, default='')
    parser.add_argument('--pos-weight', type=float, default=9.)

    parser.add_argument('--no-pretrained', dest='load_pretrained', action='store_false', default=True)
    parser.add_argument('--no-ema', action='store_true', default=False)
    parser.add_argument('--no-swa', action='store_true', default=False)
    parser.add_argument('--swa-start', type=int, default=250)
    parser.add_argument('--swa-every', type=int, default=50)
    parser.add_argument('--early-stopping-start', type=float, default=1.0)

    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--checkpoint-hist', type=int, default=5)
    parser.add_argument('--eval-metric', type=str, default='dice')
    parser.add_argument('--no-amp', action='store_true', default=False)
    parser.add_argument('--save-pred', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser()
    print('a')
    parser.add_argument("--local_rank", type=int, default=1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device
    
    # checkpoint list
    checkpoint_list = ['epoch25',
                       'epoch26',
                       'epoch27',
                       'epoch28',
                       'epoch29',]

    df = pd.read_csv('bbox_kfold_df.csv')
    df = df.loc[df['fold']==0]

    bbox_image_id_list = df['StudyInstanceUID'].values
    bbox_slice_num_list = df['slice_number'].values

    bbox_dict = {}
    for i in range(len(bbox_image_id_list)):
        xmin = df['x'].values[i]
        ymin = df['y'].values[i]
        xmax = xmin + df['width'].values[i]
        ymax = ymin + df['height'].values[i]
        bbox_dict[str(bbox_image_id_list[i])+'/'+ str(bbox_slice_num_list[i])] = [xmin, ymin, xmax, ymax]
    image_list_valid = []

    for i in range(len(bbox_image_id_list)):
        sorted_image_list = '../data/train_images/' + str(bbox_image_id_list[i]) + '/' + str(bbox_slice_num_list[i]) + '.dcm'
        image_list_valid.append(sorted_image_list)
        
    print(len(image_list_valid))

    # hyperparameters
    batch_size = 16
    image_size = 512
    criterion = nn.L1Loss().to(args.device)

    # start validation
    for idx, ckp in enumerate(checkpoint_list):

        # build model
        model = efficientnet()
        print(model.load_state_dict(torch.load('../src/bbox_weights/'+ckp)))
        
        model = model.to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        model.eval()

        pred_bbox = np.zeros((len(df),4),dtype=np.float32)

        # iterator for validation
        datagen = PEDataset(df = df, bbox_dict=bbox_dict, image_list=image_list_valid, target_size=image_size)
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        losses = AverageMeter()
        print(f'ckp : {ckp}' + f' {idx/len(checkpoint_list) * 100:.4f}% completed')
        for i, (images, labels) in enumerate(generator):
            with torch.no_grad():
                start = i*batch_size
                end = start+batch_size
                if i == len(generator)-1:
                    end = len(generator.dataset)
                images = images.to(args.device)
                labels = labels.to(args.device)
                logits = model(images)
                
                loss = criterion(logits, labels)
                losses.update(loss.item(), images.size(0))
                pred_bbox[start:end] = np.squeeze(logits.cpu().data.numpy())
        print('loss:{}'.format(losses.avg), flush=True)

if __name__ == "__main__":
    main()
