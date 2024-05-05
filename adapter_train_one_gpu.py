# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from dataset.dataset import NpyBoxDataset, NpyScribbleDataset
import wandb
from build_model import build_model
from matplotlib import pyplot as plt
import argparse
# %%

def Config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_root", type=str, default="/ai/data/data/cvpr24-medsam/data_npy",
        help="Path to the npy data root."
    )
    parser.add_argument(
        "-pretrained_checkpoint", type=str, default="work_dir/LiteMedSAM/lite_medsam.pth",
        help="Path to the pretrained Lite-MedSAM checkpoint."
    )
    parser.add_argument(
        "-train_mode", type=str, default="boxes",
        help="Path to the pretrained Lite-MedSAM checkpoint."
    )
    parser.add_argument(
        "-resume", type=str, default='None',
        help="Path to the checkpoint to continue training."
    )
    parser.add_argument("-tag", help='tag of experiment',default=None)
    # parser.add_argument(
    #     "-work_dir", type=str, default="./workdir",
    #     help="Path to the working directory where checkpoints and logs will be saved."
    # )
    parser.add_argument("-wm", "--wandb_mode", default="offline")
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-mod', type=str, default='None', help='mod type:sam_adpt,sam_lora,sam_adalora')
    parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=None , help='crop volume depth')

    parser.add_argument(
        "-num_epochs", type=int, default=10,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "-batch_size", type=int, default=4,
        help="Batch size."
    )
    parser.add_argument(
        "-num_workers", type=int, default=8,
        help="Number of workers for dataloader."
    )
    parser.add_argument(
        "-device", type=str, default="cuda:0",
        help="Device to train on."
    )
    parser.add_argument(
        "-bbox_shift", type=int, default=5,
        help="Perturbation to bounding box coordinates during training."
    )
    parser.add_argument(
        "-lr", type=float, default=0.00005,
        help="Learning rate."
    )
    parser.add_argument(
        "-weight_decay", type=float, default=0.01,
        help="Weight decay."
    )
    parser.add_argument(
        "-iou_loss_weight", type=float, default=1.0,
        help="Weight of IoU loss."
    )
    parser.add_argument(
        "-seg_loss_weight", type=float, default=1.0,
        help="Weight of segmentation loss."
    )
    parser.add_argument(
        "-ce_loss_weight", type=float, default=1.0,
        help="Weight of cross entropy loss."
    )
    args = parser.parse_args()
    tag = f"{args.train_mode}_{args.mod}_{args.tag}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    wandb.init(project=f"CVPR24_MedSAM_{args.train_mode}", name=tag,
                   config=args, mode=args.wandb_mode)
    return args,tag
# %%


# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)



class SAMIL:
    def __init__(self,args,tag):
        self.args = args
        self.work_dir = f"workdir/{tag}"
        self.data_root = args.data_root
        self.pretrained_checkpoint = args.pretrained_checkpoint
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device
        self.bbox_shift = args.bbox_shift
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.iou_loss_weight = args.iou_loss_weight
        self.seg_loss_weight = args.seg_loss_weight
        self.ce_loss_weight = args.ce_loss_weight
        self.checkpoint = args.resume
        self.train_mode = args.train_mode 
        self.dataset = NpyBoxDataset if self.train_mode == "boxes" else NpyScribbleDataset
        self.model = build_model(self.args)
        self.set_model()

        makedirs(self.work_dir, exist_ok=True)
        
# %%
    def set_model(self):
        if self.pretrained_checkpoint is not None:
            if isfile(self.pretrained_checkpoint):
                print(f"Finetuning with pretrained weights {self.pretrained_checkpoint}")
                ckpt = torch.load(
                    self.pretrained_checkpoint,
                    map_location="cpu"
                )
                # medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
                for name, param in self.model.named_parameters():
                        
                    if name in ckpt:
                        param_shape = ckpt[name].shape
                        if param.shape == param_shape:
                            param.data.copy_(ckpt[name])
                        else:
                            print(f"Shape mismatch for parameter '{name}'. Skipping...")
                    else:
                        print(f"Parameter '{name}' not found in the checkpoint. Skipping...")

                
            else:
                print(f"Pretained weights {self.pretrained_checkpoint} not found, training from scratch")

        self.model = self.model.to(self.device)
        
        if self.args.mod == 'sam_adpt':
                for n, value in self.model.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
        elif self.args.mod == 'sam_lora' or self.args.mod == 'sam_adalora':
            from models.common import loralib as lora
            lora.mark_only_lora_as_trainable(self.model.image_encoder)
            if self.args.mod == 'sam_adalora':
                # Initialize the RankAllocator 
                self.rankallocator = lora.RankAllocator(
                    self.model.image_encoder, lora_r=4, target_rank=8,
                    init_warmup=500, final_warmup=1500, mask_interval=10, 
                    total_step=3000, beta1=0.85, beta2=0.85, 
                )
        else:
            for n, value in self.model.image_encoder.named_parameters(): 
                value.requires_grad = True
        print(f"MedSAM Lite size: {sum(p.numel() for p in self.model.parameters())}")

    def train(self):

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,
            patience=5,
            cooldown=0
        )
        seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        iou_loss = nn.MSELoss(reduction='mean')
        # %%
        # if self.train_mode == "Box":
        #     train_dataset = NpyBoxDataset(data_root=self.data_root, data_aug=True)
        # else:
        #     train_dataset = NpyBoxDataset(data_root=self.data_root, data_aug=True)
        train_dataset = self.dataset(data_root=self.data_root, num_each_epoch=100000 ,data_aug=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        if self.checkpoint and isfile(self.checkpoint):
            print(f"Resuming from checkpoint {self.checkpoint}")
            checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["loss"]
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_loss = 1e10
        if self.args.mod == 'sam_adalora':
            ind = 0
        # %%
        train_losses = []
        boxes = None
        masks = None
        self.model.train()
        for epoch in range(start_epoch + 1, self.num_epochs):
            epoch_loss = [1e10 for _ in range(len(train_loader))]
            epoch_start_time = time()
            pbar = tqdm(train_loader)
            for step, batch in enumerate(pbar):
                if self.args.mod == 'sam_adalora':
                    ind += 1
                image = batch["image"]
                gt2D = batch["gt2D"]
                optimizer.zero_grad()
                if self.train_mode == "boxes":
                    boxes = batch["bboxes"]
                    boxes = boxes.to(self.device)
                else:
                    masks = batch['scribbles']
                    masks = masks.to(self.device)
                    
                
                image, gt2D = image.to(self.device), gt2D.to(self.device)
                logits_pred, iou_pred = self.model(image, boxes,masks)
                l_seg = seg_loss(logits_pred, gt2D)
                l_ce = ce_loss(logits_pred, gt2D.float())
                #mask_loss = l_seg + l_ce
                mask_loss = self.seg_loss_weight * l_seg + self.ce_loss_weight * l_ce
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                l_iou = iou_loss(iou_pred, iou_gt)
                #loss = mask_loss + l_iou
                loss = mask_loss + self.iou_loss_weight * l_iou
                epoch_loss[step] = loss.item()
                # loss.backward()

                if self.args.mod == 'sam_adalora':
                    from models.common import loralib as lora
                    (loss+lora.compute_orth_regu(self.model, regu_weight=0.1)).backward()
                    optimizer.step()
                    self.rankallocator.update_and_mask(self.model, ind)
                else:
                    loss.backward()
                    optimizer.step()



                # optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

            epoch_end_time = time()
            epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
            train_losses.append(epoch_loss_reduced)
            lr_scheduler.step(epoch_loss_reduced)
            model_weights = self.model.state_dict()
            wandb.log({'loss': epoch_loss_reduced})
            checkpoint = {
                "model": model_weights,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss_reduced,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, join(self.work_dir, "medsam_lite_latest.pth"))
            if epoch_loss_reduced < best_loss:
                print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
                best_loss = epoch_loss_reduced
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, join(self.work_dir, "medsam_lite_best.pth"))

            epoch_loss_reduced = 1e10
            # %% plot loss
            plt.plot(train_losses)
            plt.title("Dice + Binary Cross Entropy + IoU Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(self.work_dir, "train_loss.png"))
            plt.close()


if __name__ == "__main__":
    args,tag = Config()
    trainer = SAMIL(args,tag)
    trainer.train()

    
