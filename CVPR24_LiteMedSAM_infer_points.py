from glob import glob
from os import listdir, makedirs
from os.path import join, isfile, basename
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp
from dataset.utils import resize_longest_side,pad_image,show_mask
import argparse
from build_model import build_model

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='/ai/data/data/cvpr24-medsam/validation-scribbles/validation_scribbles-001-k41c',
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='/demo_scribble/segs/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="work_dir/LiteMedSAM/medsam_lite_scribble.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument('-mod', type=str, default='None', help='mod type:seg,cls,val_ad')
parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
parser.add_argument('-thd', type=bool, default=False , help='3d or not')
parser.add_argument('-chunk', type=int, default=None , help='crop volume depth')
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=1,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='.',
    help='directory to save the overlay image'
)

args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
image_size = 256

medsam_lite_model = build_model(args,False)
lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
medsam_lite_model.to(device)
medsam_lite_model.eval()



@torch.no_grad()
def medsam_inference(medsam_model, img_embed, point_coords, new_size, original_size):
    point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=img_embed.device)
    point_labels = torch.ones((point_coords.shape[0],point_coords.shape[1]))
    points = (point_coords, point_labels)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=points,
        boxes=None,
        masks=None,
    )
    low_res_masks, iou_predictions = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    # low_res_pred = torch.sigmoid(low_res_pred)
    # low_res_pred = low_res_pred.squeeze().cpu().numpy()
    # medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    low_res_pred = medsam_model.postprocess_masks(low_res_masks, new_size, original_size)
    # low_res_pred = torch.sigmoid(low_res_pred)  
    medsam_seg = low_res_pred.squeeze().cpu()

    return medsam_seg, iou_predictions



# %%
def MedSAM_infer_npz(gt_path_file):
    npz_name = basename(gt_path_file)
    npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    scribble = npz_data['scribbles']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## MedSAM Lite preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)


    label_ids = np.unique(scribble[(scribble != 0) & (scribble != 1000)])
    scribbles_output = []
    for label_id in label_ids:
        
        point_coords_each_label = np.argwhere(scribble == label_id)[None]
        
        medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, point_coords_each_label, (newh, neww), (H, W))
        scribbles_output.append(medsam_mask)
    if len(scribbles_output) > 1:
        scribbles_output = torch.stack(scribbles_output)
    else:
        scribbles_output = scribbles_output[0]
    

    if len(scribbles_output.shape) == 2:
        # medsam_mask = torch.sigmoid(medsam_mask ) 
        scribbles_output = (scribbles_output > 0).int()
    else: 
        new_die = torch.zeros([1,H, W])
        scribbles_output = torch.cat([new_die,scribbles_output])
        # medsam_mask += 1
        scribbles_output = torch.argmax(scribbles_output, axis=0)
    
        # if np.max(label_ids) > scribbles_output.max():
        #     print("label ID is not continue")
        #     for idx, label_id in enumerate(label_ids):
        #         scribbles_output[scribbles_output == idx+1] = label_id


    segs = scribbles_output.numpy()
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs
    )

    # visualize image, mask and bounding box
    if save_overlay:
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[2].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("Scribbled Image")
        ax[2].set_title(f"Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i, label_id in enumerate(label_ids):
            color = np.random.rand(3)
            show_mask((scribble==label_id).astype(np.uint8), ax[1], mask_color=color)
            show_mask((segs == label_id).astype(np.uint8), ax[2], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'))#, dpi=300)
        plt.close()

if __name__ == '__main__':
    num_workers = 1

    # mp.set_start_method('spawn')
    # with mp.Pool(processes=num_workers) as pool:
    #     with tqdm(total=len(gt_path_files)) as pbar:
    #         for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_npz, gt_path_files))):
    #             pbar.update()

    # with tqdm(total=len(gt_path_files)) as pbar:
    #     for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_npz, gt_path_files))):
    #         pbar.update()
    for img_npz_file in tqdm(gt_path_files):
        MedSAM_infer_npz(img_npz_file)

    
# %%
