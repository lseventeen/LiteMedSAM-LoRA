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
from utils.loss import loss_masks
import torch.nn.functional as F
from models.efficientvit.sam import efficientvit_sam_l0,efficientvit_sam_l1,efficientvit_sam_l2
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
    parser.add_argument(
        "-num_masks", type=int, default=16,
        help="set the number of mask for each image"
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
        "-num_epochs", type=int, default=20,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "-num_each_epoch", type=int, default=100000,
        help="Number of data in each epochs to train."
    )
    
    parser.add_argument(
        "-batch_size", type=int, default=4,
        help="Batch size."
    )
    parser.add_argument(
        "-num_workers", type=int, default=16,
        help="Number of workers for dataloader."
    )
    parser.add_argument(
        "-device", type=str, default="cuda:0",
        help="Device to train on."
    )
    parser.add_argument(
        "-bbox_shift", type=int, default=10,
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
        self.num_each_epoch = args.num_each_epoch
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
        self.num_masks = args.num_masks 
        self.amp = "fp16"
        self.grad_clip = 2
        self.dataset = NpyBoxDataset if self.train_mode == "boxes" else NpyScribbleDataset
        # self.model = build_model(self.args)
        self.model = efficientvit_sam_l0()
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.9,
            patience=5,
            cooldown=0
        )
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_loss = nn.MSELoss(reduction='mean')
        # %%
        # if self.train_mode == "Box":
        #     train_dataset = NpyBoxDataset(data_root=self.data_root, data_aug=True)
        # else:
        #     train_dataset = NpyBoxDataset(data_root=self.data_root, data_aug=True)
        train_dataset = self.dataset(data_root=self.data_root, num_each_epoch=self.num_each_epoch ,image_size = 512,num_masks = self.num_masks,data_aug=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        if self.checkpoint and isfile(self.checkpoint):
            print(f"Resuming from checkpoint {self.checkpoint}")
            checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["loss"]
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_loss = 1e10
        if self.args.mod == 'sam_adalora':
            ind = 0
        self.train_losses = []
        self.model.train()
        for epoch in range(start_epoch + 1, self.num_epochs):
            self.optimizer.zero_grad()
            self._train_one_epoch(epoch)
            model_weights = self.model.state_dict()
            # self.after_step()
            wandb.log({'loss': self.epoch_loss_reduced})
            checkpoint = {
                "model": model_weights,
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "loss": self.epoch_loss_reduced,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, join(self.work_dir, "medsam_lite_latest.pth"))
            if self.epoch_loss_reduced < best_loss:
                print(f"New best loss: {best_loss:.4f} -> {self.epoch_loss_reduced:.4f}")
                best_loss = self.epoch_loss_reduced
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, join(self.work_dir, "medsam_lite_best.pth"))

            # %% plot loss
            plt.plot(self.train_losses)
            plt.title("Dice + Binary Cross Entropy + IoU Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(self.work_dir, "train_loss.png"))
            plt.close()


       
    def _train_one_epoch(self,epoch):
        # %%
        
        epoch_loss = [1e10 for _ in range(len(self.train_loader))]
        self.epoch_start_time = time()
        pbar = tqdm(self.train_loader)

        for step, batch in enumerate(pbar):
            if self.args.mod == 'sam_adalora':
                ind += 1
            image = batch["image"]
            gt = batch["gt2D"]
            image, gt = image.to(self.device), gt.to(self.device)
            self.optimizer.zero_grad()
            if self.train_mode == "boxes":
                boxes = batch["bboxes"]
                boxes = boxes.to(self.device)
            else:
                masks_input = batch['mask_inputs']
                masks_input = masks_input.to(self.device)

            batched_input = []
            for b_i in range(len(image)):
                dict_input = dict()
                dict_input["image"] = image[b_i]
                if self.train_mode == "boxes":
                    dict_input["boxes"] = boxes[b_i]
                else:
                    dict_input["mask_inputs"] = masks_input[b_i] 
    
                batched_input.append(dict_input)
                
            
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                if random.random() >= 0.5:
                    output, iou_predictions = self.model(batched_input, multimask_output=True)
                else:
                    output, iou_predictions = self.model(batched_input, multimask_output=False)
                    # output = self.model.postprocess_masks(output,)
                gt = gt.reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)
                loss_list = []
                for i in range(output.shape[2]):
                    output_i = (
                        F.interpolate(output[:, :, i], size=(image.shape[2], image.shape[3]), mode="bilinear")
                        .reshape(-1, image.shape[2], image.shape[3])
                        .unsqueeze(1)
                    )
                    # output_i = output[:, :, i].reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)
                    
                
                    loss_mask_i, loss_dice_i = loss_masks(output_i, gt, len(output_i), mode="none")
                    loss_i = loss_mask_i + loss_dice_i
                    loss_list.append(loss_i)
                loss = torch.stack(loss_list, -1)

                min_indices = torch.argmin(loss, dim=1)
                mask = torch.zeros_like(loss, device=loss.device)
                mask.scatter_(1, min_indices.unsqueeze(1), 1)

                loss = (loss * mask).mean() * loss.shape[-1]
                # logits_pred, iou_pred = self.model(batched_input,False)
                
                # l_seg = self.seg_loss(logits_pred, gt2D)
                # l_ce = self.ce_loss(logits_pred, gt2D.float())
                # #mask_loss = l_seg + l_ce
                # loss = self.seg_loss_weight * l_seg + self.ce_loss_weight * l_ce
                # iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                # l_iou = self.iou_loss(iou_pred, iou_gt)
                #loss = mask_loss + l_iou
                # loss = mask_loss + self.iou_loss_weight * l_iou
                epoch_loss[step] = loss.item()
                # loss.backward()

                if self.args.mod == 'sam_adalora':
                    from models.common import loralib as lora
                    (loss+lora.compute_orth_regu(self.model, regu_weight=0.1)).backward()
                    self.optimizer.step()
                    self.rankallocator.update_and_mask(self.model, ind)
                else:
                    # loss.backward()
                    self.scaler.scale(loss).backward()
                    # self.optimizer.step()


            self.scaler.unscale_(self.optimizer)
        # gradient clip
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
            # update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
        self.epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        self.train_losses.append(self.epoch_loss_reduced)
        self.lr_scheduler.step(self.epoch_loss_reduced)
    @property
    def enable_amp(self) -> bool:
        return self.amp != "fp32"
    @property
    def amp_dtype(self) -> torch.dtype:
        if self.amp == "fp16":
            return torch.float16
        elif self.amp == "bf16":
            return torch.bfloat16
        else:
            return torch.float32
        

if __name__ == "__main__":
    args,tag = Config()
    trainer = SAMIL(args,tag)
    trainer.train()

    
