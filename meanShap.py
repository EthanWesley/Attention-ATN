import os
from tqdm import tqdm, trange
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
from collections import defaultdict
import time
import argparse
import numpy as np
from PIL import Image
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from data_function7 import MedData_train,MedData_test
import torch.nn.functional as F
import random
import wandb  ##================================a
import os
import shap

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_data_dir = r'/data/sjwlab/weicy/dataset/ds_f' #/home/clks/weicy/dataset/ds_2



ckpt_path= 'your_checkpoint.pt'
output_dir = '/your_output_dir'

target = 'A'
model_depth = 34
hp_dropout = 0



from models.three_d.resnet3d_sa import generate_model,ResNetWithSelfAttention
from models.three_d.attention import AttentionFusionModel
modal_t1 = generate_model(model_depth, n_input_channels=1, n_classes=2)
modal_t1_sa = ResNetWithSelfAttention(modal_t1)
modal_t2 = generate_model(model_depth, n_input_channels=1, n_classes=2)
modal_t2_sa=  ResNetWithSelfAttention(modal_t2)
model = AttentionFusionModel(modal_t1_sa, modal_t2_sa, clinical_feature_dim=7, dropout=hp_dropout)

model = torch.nn.DataParallel(model, device_ids=devicess)

print("load model:", ckpt_path)
ckpt = torch.load(os.path.join(output_dir, ckpt_path),
                  map_location=lambda storage, loc: storage)

model.load_state_dict(ckpt["model"])



# scheduler.load_state_dict(ckpt["scheduler"])
elapsed_epochs = ckpt["epoch"]
epoch = ckpt["epoch"]
print("epoch",elapsed_epochs)

model.to(device)

train_dataset = MedData_test(os.path.join(source_data_dir, "train_data.xlsx"),
                              os.path.join(source_data_dir, "train"))
train_loader = DataLoader(train_dataset.testing_set,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=False,
                         drop_last=False)
val_dataset = MedData_test(os.path.join(source_data_dir, "val_data.xlsx"),
                              os.path.join(source_data_dir, "val"))
val_loader = DataLoader(val_dataset.testing_set,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=False,
                         drop_last=False)
test_dataset = MedData_test(os.path.join(source_data_dir, "test_data.xlsx"),
                            os.path.join(source_data_dir, "test"))
test_loader = DataLoader(test_dataset.testing_set,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=False,
                         drop_last=False)


# plot_and_save_shap_heatmaps(x_t2, shap_values_T2, csfID, 'T2')
for i, train_batch in enumerate(train_loader):
    train_x_t1 = train_batch['T1']['data'].to(device)
    train_x_t2 = train_batch['T2']['data'].to(device)
    train_clinical_features = train_batch['clinical'].to(device)
    train_y = train_batch[target].to(device)

explainer = shap.GradientExplainer(model,
                                     [train_x_t1, train_x_t2, train_clinical_features],batch_size=15)          


import pickle
from tqdm import trange
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib.cm as cm

for i, test_batch in tqdm(enumerate(test_loader)):
    test_x_t1 = test_batch['T1']['data'].to(device)
    test_x_t2 = test_batch['T2']['data'].to(device)
    test_clinical_features = test_batch['clinical'].to(device)
    test_y = test_batch[target].to(device)

    shap_values = explainer.shap_values([test_x_t1, test_x_t2, test_clinical_features]) 

    #print(shap_values[0].shape)
    print(test_x_t1.cpu().numpy().shape)
    # break

    reshaped_data_T1 = shap_values[1][0].transpose(4, 2, 3, 1, 0).reshape(40, 200, 200)

    with open('data.pkl', 'wb') as f:
        pickle.dump(reshaped_data_T1, f)
    
