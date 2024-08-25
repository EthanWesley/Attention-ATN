import os
from tqdm import tqdm, trange
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda:0")
from collections import defaultdict
import time
import argparse
import numpy as np
from PIL import Image

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
from hparam_gridsearch import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from data_function import MedData_train,MedData_test
import torch.nn.functional as F

import wandb  ##================================a
import os
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)


source_data_dir = hp.source_data_dir


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    return parser

def main(hp_weightdecay,hp_dropout,target):
    #wandb.init()
    model_depth = 34
    best_val_auc = 0.7
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    args.output_dir = args.output_dir + "_"+ target + "_wd" + str(hp_weightdecay)+ "_dr" + str(hp_dropout)
    output_dir_pt = args.output_dir+ "_pt"
    os.makedirs(output_dir_pt, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

        
    os.makedirs(args.output_dir, exist_ok=True)



    from models.three_d.resnet3d_sa import generate_model,ResNetWithSelfAttention
    from models.three_d.attention import AttentionFusionModel
    modal_t1 = generate_model(model_depth, n_input_channels=1, n_classes=2)
    modal_t1_sa = ResNetWithSelfAttention(modal_t1)
    modal_t2 = generate_model(model_depth, n_input_channels=1, n_classes=2)
    modal_t2_sa=  ResNetWithSelfAttention(modal_t2)
    model = AttentionFusionModel(modal_t1_sa, modal_t2_sa, clinical_feature_dim=7, dropout=hp_dropout)
    #

    model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=hp_weightdecay)

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from loss_function import Classification_Loss
    criterion = Classification_Loss().cuda()

    writer = SummaryWriter(args.output_dir)

    ##########################################
    # data
    ##########################################

    train_dataset = MedData_train(os.path.join(source_data_dir, "train_data.xlsx"),
                                  os.path.join(source_data_dir, "train"))
    train_loader = DataLoader(train_dataset.training_set,
                              batch_size=hp.batch_size,
                              shuffle=True,
                              num_workers=hp.num_worker,
                              pin_memory=False,
                              drop_last=True)
    val_dataset = MedData_test(os.path.join(source_data_dir, "val_data.xlsx"),
                                  os.path.join(source_data_dir, "val"))
    val_loader = DataLoader(val_dataset.testing_set,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    test_dataset = MedData_test(os.path.join(source_data_dir, "test_data.xlsx"),
                                os.path.join(source_data_dir, "test"))
    test_loader = DataLoader(test_dataset.testing_set,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)


    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    for epoch in trange(1, epochs + 1):
        #print("epoch:" + str(epoch))
        if not model.training:
            model.train()
        epoch += elapsed_epochs
        num_iters = 0
        total_loss = 0

        gts = []
        predicts = []
        train_prob =[]
        for i, batch in enumerate(train_loader):

            if hp.debug:
                if i >= 1:
                    break

            #print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()

            x_t1 = batch['T1']['data']
            x_t2 = batch['T2']['data']
            clinical_features = batch['clinical']
            y = batch[target]


            x_t1 = x_t1.type(torch.FloatTensor).cuda()
            x_t2 = x_t2.type(torch.FloatTensor).cuda()
            y = y.type(torch.LongTensor).cuda()

            if hp.mode == '2d':
                x = x.squeeze(-1)
                x = x[:, :1, :, :]

            outputs = model(x_t1, x_t2, clinical_features)

            outputs_prob = F.softmax(outputs, dim=1)
            outputs_label = outputs_prob.argmax(dim=1)
            outputs_prob_1 = outputs_prob[:,1]

            loss = criterion(outputs, y, model)
            total_loss += loss
            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1
            predicts.append(outputs_label.cpu().detach().numpy())
            gts.append(y.cpu().detach().numpy())
            train_prob.append(outputs_prob_1.cpu().detach().numpy())
            x_t1 = 0
            x_t2 = 0
            del x_t1
            del x_t2
            torch.cuda.empty_cache()

        predicts = np.concatenate(predicts).flatten().astype(np.int16)
        gts = np.concatenate(gts).flatten().astype(np.int16)
        train_prob = np.concatenate(train_prob).flatten().astype(np.float32)
        cm = metrics.confusion_matrix(gts,predicts)

        TN = cm[0, 0]   
        FP = cm[0, 1] 
        FN = cm[1, 0]   

        #  NPV
        train_NPV = TN / (TN + FN)

        #  Specificity
        train_specificity = TN / (TN + FP)

        acc = metrics.accuracy_score( gts,predicts)
        recall = metrics.recall_score(gts,predicts)
        f1 = metrics.f1_score(gts,predicts)
        auc = metrics.roc_auc_score(gts, train_prob)
        precision_score = metrics.precision_score(gts,predicts, zero_division=0)
        train_average_loss = total_loss/len(train_loader)
        print('epoch',epoch,'\n    matrix \n',cm, "\n    acc",acc,"auc",auc,'Average_Loss',train_average_loss)
        writer.add_scalar('Training/Average_Loss', train_average_loss, epoch)
        writer.add_scalar('Training/acc', acc, epoch)
        writer.add_scalar('Training/recall(Sensitivity)', recall, epoch)
        writer.add_scalar('Training/f1', f1, epoch)
        writer.add_scalar('Training/PPV(precision_score)', precision_score, epoch)
        writer.add_scalar('Training/auc', auc, epoch)
        writer.add_scalar('Training/NPV', train_NPV, epoch)
        writer.add_scalar('Training/specificity', train_specificity, epoch)
        # wandb.log({
        #     'epoch': epoch,
        #     'Train/train_acc': acc,
        #     'Train/train_loss': train_average_loss,
        #     'Train/train_recall(Sensitivity)': recall,
        #     'Train/train_f1': f1,
        #     'Train/train_precision_score(PPV)': precision_score,
        #     'Train/train_auc': auc,
        #     'Train/train_NPV': train_NPV,
        #     'Train/train_specificity': train_specificity,
        # })
        scheduler.step()

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0: #args.epochs_per_checkpoint


            model.eval()
            val_predicts = []
            val_gts = []
            val_prob = []

            val_total_loss = 0
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    x_t1 = batch['T1']['data'].to(device)
                    x_t2 = batch['T2']['data'].to(device)
                    clinical_features = batch['clinical']
                    y = batch[target].to(device)

                    outputs = model(x_t1, x_t2, clinical_features)


                    outputs_prob = F.softmax(outputs, dim=1)
                    outputs_label = outputs_prob.argmax(dim=1)
                    outputs_prob_1 = outputs_prob[:, 1]

                    loss = criterion(outputs, y, model)

                    val_total_loss += loss

                    val_predicts.append(outputs_label.cpu().detach().numpy())
                    val_gts.append(y.cpu().detach().numpy())
                    val_prob.append(outputs_prob_1.cpu().detach().numpy())
                    ###https://blog.csdn.net/weixin_44826203/article/details/130401177
                    x_t1 = 0
                    x_t2 = 0
                    del x_t1
                    del x_t2
                    torch.cuda.empty_cache()
                # torch.cuda.empty_cache()

            val_predicts = np.concatenate(val_predicts).flatten().astype(np.int16)
            val_gts = np.concatenate(val_gts).flatten().astype(np.int16)
            val_prob = np.concatenate(val_prob).flatten().astype(np.float32)
            cm = metrics.confusion_matrix(val_gts,val_predicts)

            TN = cm[0, 0]  # 真阴性
            FP = cm[0, 1]  # 假阳性
            FN = cm[1, 0]  # 假阴性

            # 计算 NPV
            val_NPV = TN / (TN + FN)

            # 计算 Specificity
            val_specificity = TN / (TN + FP)
            print(f"checkpoint_{epoch:04d}.pt" + "val confusion_matrix:",
                  )
            val_acc = metrics.accuracy_score(val_gts,val_predicts)
            val_recall = metrics.recall_score(val_gts,val_predicts, zero_division=0)
            val_f1 = metrics.f1_score(val_gts,val_predicts)
            val_auc = metrics.roc_auc_score(val_gts, val_prob)
            val_precision_score = metrics.precision_score(val_gts,val_predicts, zero_division=0)
            val_average_loss = val_total_loss / len(val_loader)
            print(f"@@checkpoint_{epoch:04d}.pt@@", '\n    val_matrix\n', cm, "\n    val_acc", val_acc, "val_auc", val_auc,'Average_Loss',val_average_loss)
            # wandb.log({
            #     'epoch': epoch,
            #     'Val/val_acc': val_acc,
            #     'Val/val_loss': val_average_loss,
            #     'Val/val_recall': val_recall,
            #     'Val/val_f1': val_f1,
            #     'Val/val_precision_score': val_precision_score,
            #     'Val/val_auc': val_auc,
            #     'Val/val_NPV': val_NPV,
            #     'Val/val_specificity': val_specificity,
            # })
            writer.add_scalar('Val/Average_Loss', val_average_loss, epoch)
            writer.add_scalar('Val/acc', val_acc, epoch)
            writer.add_scalar('Val/recall(Sensitivity)', val_recall, epoch)
            writer.add_scalar('Val/f1', val_f1, epoch)
            writer.add_scalar('Val/PPV(precision_score)', val_precision_score, epoch)
            writer.add_scalar('Val/auc', val_auc, epoch)
            writer.add_scalar('Val/NPV', val_NPV, epoch)
            writer.add_scalar('Val/specificity', val_specificity, epoch)


        if epoch==epochs:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(output_dir_pt, f"checkpoint_{epoch:04d}.pt"),
            )                

if __name__ == '__main__':

    dr_list = [0.1,0.2,0.3,0.4]
    wd_list = [0.1,0.01,0.001,0.0001]

    for dr in tqdm(dr_list, desc='Dropout', leave=False):
        for wd in tqdm(wd_list, desc='Weight Decay'):
            main(hp_weightdecay=wd, hp_dropout=dr, target='A')

