from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.utils import show_box,resize_longest_side,pad_image,resize_box_to_256,show_mask,get_bbox256
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from models.tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from build_model import build_model

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='/ai/data/data/cvpr24-medsam/validation-box/imgs-011/imgs',
    # required=True,
    help='root directory of the data',
)

parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='test_demo/segs/',
    help='directory to save the prediction',
)
# parser.add_argument(
#         "-train_mode", type=str, default="boxes",
#         help="Path to the pretrained Lite-MedSAM checkpoint."
#     )
parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="workdir/medsam_lite_latest.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument('-mod', type=str, default='None', help='mod type:seg,cls,val_ad')
parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
parser.add_argument('-thd', type=bool, default=False , help='3d or not')
parser.add_argument('-chunk', type=int, default=None , help='crop volume depth')
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay/',
    help='directory to save the overlay image'
)

args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256



medsam_lite_model = build_model(args)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
medsam_lite_model.to(device)
medsam_lite_model.eval()


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou


def MedSAM_infer_npz_2D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
    # boxes256 = resize_box_to_256(boxes, original_size=(H, W))
    # medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, boxes256, (newh, neww), (H, W))
    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W))
        segs[medsam_mask>0] = idx
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
   
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
    if save_overlay:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


def MedSAM_infer_npz_3D(img_npz_file,new_size,scale = 0.75):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]
    boxes_3D = resize_boxes(boxes_3D, scale,scale)

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max+1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                pre_seg256 = resize_longest_side(pre_seg)
                if np.max(pre_seg256) > 0:
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256[None], [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        zmin = max(0, z_min-1)
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            pre_seg256 = resize_longest_side(pre_seg)
            if np.max(pre_seg256) > 0:
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256[None], [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )            

    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3DV2(img_npz_file)
        else:
            MedSAM_infer_npz_2D(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
