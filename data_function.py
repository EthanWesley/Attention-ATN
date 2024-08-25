from glob import glob
from os.path import dirname, join, basename, isfile
import sys

import pandas as pd

sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RandomBlur,
    RandomSpike,
    RandomGamma,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    RandomAnisotropy,
    Compose,
)
from pathlib import Path
import os
from hparam import hparams as hp

# MedData_train("val_data.xlsx","data/val")
class MedData_train(torch.utils.data.Dataset):
    def __init__(self, excel_file, dataset_path):

        self.dataset_path = dataset_path
        self.data_info = pd.read_excel(excel_file)
        self.subjects = []

        for _, row in self.data_info.iterrows():
            subject = tio.Subject(
                T1=tio.ScalarImage(os.path.join(dataset_path, 'T1', f"I{row['T1']}.nii.gz")),
                T2=tio.ScalarImage(os.path.join(dataset_path, 'T2', f"I{row['T2']}.nii.gz")),
                clinical=torch.tensor(row[7:14].values.astype(np.float32)),
                A=row['A'],
                T=row['T'],
                N=row['N'],
                ID=row['CSFID']
            )
            self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        # one_subject = self.training_set[0]
        # one_subject.plot()
    def transform(self):
        if hp.aug:
            training_transform = Compose([
                                            ZNormalization(),
                                            RandomBlur(std=(0, 2)),  # 增加模糊的标准差范围
                                            RandomGamma(log_gamma=(-0.5, 0.5)),  # 增加Gamma变化范围
                                            RandomNoise(mean=0, std=(0, 1)),  # 增加噪声强度范围
                                            RandomFlip(axes=('lr',), flip_probability=0.5),  # 保留左右翻转
                                            OneOf({
                                                RandomAffine(scales=(0.8, 1.2), degrees=(0, 30), translation=(0, 10)): 0.2,  # 调整仿射变换参数
                                                RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 5)): 0.2,  # 调整各向异性参数
                                                RandomBiasField(coefficients=(0.5, 1.5)): 0.2,  # 调整偏置场参数
                                                RandomSpike(num_spikes=(1, 3), intensity=(0.5, 1.5)): 0.2,  # 调整尖峰噪声参数
                                                RandomElasticDeformation(num_control_points=(7, 7, 7), max_displacement=(15, 15, 15)): 0.2  # 添加弹性变形
                                            }),
                                            ZNormalization(),
                                        ])


        else:
            training_transform = Compose([
                ZNormalization(),
            ])            


        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, excel_file, dataset_path):

        self.dataset_path = dataset_path
        self.data_info = pd.read_excel(excel_file)
        self.subjects = []

        for _, row in self.data_info.iterrows():
            subject = tio.Subject(
                T1=tio.ScalarImage(os.path.join(dataset_path,'T1', f"I{row['T1']}.nii.gz")),
                T2=tio.ScalarImage(os.path.join(dataset_path,'T2', f"I{row['T2']}.nii.gz")),
                clinical=torch.tensor(row[7:14].values.astype(np.float32)),
                A=row['A'],
                T=row['T'],
                N=row['N'],
                ID=row['CSFID']
            )
            self.subjects.append(subject)

        self.transforms = self.transform()

        self.testing_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        # one_subject = self.training_set[0]
        # one_subject.plot()

    def transform(self):

        testing_transform = Compose([
            ZNormalization(),
        ])
        return testing_transform
    def __len__(self):
        return len(self.testing_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        subject_idx = self.subjects[idx]
        return subject_idx
