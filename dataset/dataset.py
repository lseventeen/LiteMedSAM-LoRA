import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
import os
from os.path import join
import random
import torch
import cv2
from visual_sampler.sampler import build_shape_sampler
from visual_sampler.config import cfg
from dataset.utils import resize_longest_side,pad_image
# from monai.transforms import Compose, RandomAffine, RandomGamma


class NpyBoxDataset(Dataset): 
    def __init__(self, data_root, num_each_epoch = 100000,image_size=256, bbox_shift=5, num_masks = 16, data_aug=True):
        self.data_root = data_root
        self.data_dict = {}
        data_mode = os.listdir(data_root)
        

        for i in data_mode:
            # self.data_list.append({i: [f for f in os.listdir(join(data_root,i,"gts")) if isfile(join(data_root,i,"gts", f)) and  f.endswith(".npy")]})
            # self.data_list.append({i: os.listdir(join(data_root,i,"gts"))})
            self.data_dict[i]=os.listdir(join(data_root,i,"gts"))
            print(i,len(os.listdir(join(data_root,i,"gts"))))
            
        # self.gt_path = join(data_root, 'gts')
        # self.img_path = join(data_root, 'imgs')
        # self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        # self.gt_path_files = [
        #     file for file in self.gt_path_files
        #     if isfile(join(self.img_path, basename(file)))
        # ]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.num_each_epoch = num_each_epoch
        self.num_masks = num_masks 
    
    def __len__(self):
        return self.num_each_epoch

    def __getitem__(self, index):
        if random.random() < 0.4:
            data_mode = "CT"
        elif random.random() < 0.5:
            data_mode = "MR"
        else:
            data_mode  = random.choice(list(self.data_dict.keys()))
        # print(data_mode)
        img_name = random.choice(self.data_dict.get(data_mode))
        
        # assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]



        # print(f"{data_mode}:{img_name}")
        # img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_3c = np.load(join(self.data_root,data_mode,"imgs", img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_resize = resize_longest_side(img_3c,self.target_length)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = pad_image(img_resize,target_size=self.image_size) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(join(self.data_root,data_mode,"gts", img_name), 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = pad_image(gt,target_size=self.image_size) # (256, 256)
        label_ids = np.unique(gt)[1:]
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt = np.ascontiguousarray(np.flip(gt, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt = np.ascontiguousarray(np.flip(gt, axis=-2))
                # print('DA with flip upside down')
            
        gts = []
        boxes = []
        for i in range(self.num_masks):
            try:
                gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
            except:
                print(img_name, 'label_ids.tolist()', label_ids.tolist())
                gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
            # add data augmentation: random fliplr and random flipud
            
            gt2D = np.uint8(gt2D > 0)
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            box = np.array([x_min, y_min, x_max, y_max])
            gts.append(gt2D)
            boxes.append(box)
        gts = np.stack(gts,axis=0)
        boxes = np.stack(boxes,axis=0)
        # masks = 
        



        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gts).float(),
            "bboxes": torch.tensor(boxes).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    

class NpyScribbleDataset(NpyBoxDataset): 
    def __init__(self, data_root, num_each_epoch = 100000, image_size=256, num_masks = 16, max_num_point = 1000000,data_aug=True):
        super().__init__(data_root, num_each_epoch,image_size, num_masks = num_masks, data_aug = data_aug)
        self.max_num = max_num_point
        self.shape_sampler = build_shape_sampler(cfg)
    

    def __getitem__(self, index):
        if random.random() < 0.4:
            data_mode = "CT"
        elif random.random() < 0.5:
            data_mode = "MR"
        else:
            data_mode  = random.choice(list(self.data_dict.keys()))
        # print(data_mode)
        img_name = random.choice(self.data_dict.get(data_mode))
        
        img_3c = np.load(join(self.data_root,data_mode,"imgs", img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_resize = resize_longest_side(img_3c,self.target_length)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = pad_image(img_resize,target_size=self.image_size) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(join(self.data_root,data_mode,"gts", img_name), 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = pad_image(gt,target_size=self.image_size) # (256, 256)
        label_ids = np.unique(gt)[1:]

        gts = []
        # mask_inputs = []
        point_coords = []
        point_coords_num = []
        # point_labels = []
        
        for _ in range(self.num_masks):
        # for i in label_ids:
            try:
                gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)

                # gt2D = np.uint8(gt == i) # only one label, (256, 256)
            except:
                print(img_name, 'label_ids.tolist()', label_ids)
                gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
            # add data augmentation: random fliplr and random flipud
            if self.data_aug:
                if random.random() > 0.5:
                    img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                    # print('DA with flip left right')
                if random.random() > 0.5:
                    img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                    # print('DA with flip upside down')
            gt2D = np.uint8(gt2D > 0)
            scribbles = (self.shape_sampler(gt2D) * 1)
            point_coords_each_label = np.argwhere(scribbles[0] == 1)
            point_coords_num.append(point_coords_each_label.shape[1])

            point_coords_each_label_num = self.max_num - point_coords_each_label.shape[1]
            pad_width = [(0, 0), (0, point_coords_each_label_num)]
            
            # 执行填充
            point_coords_each_label = np.pad(point_coords_each_label, pad_width, mode='constant', constant_values=0)
            
            gts.append(gt2D)
            point_coords.append(point_coords_each_label)
            
            # point_labels.append(i)
      

        gts = np.stack(gts,axis=0)
        point_coords = np.stack(point_coords,axis=0).transpose(0,2,1)
        point_coords_num = np.stack(point_coords_num,axis=0)
        # point_labels = np.stack(point_labels,axis=0)
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gts).float(),
            "point_coords": torch.tensor(point_coords).float(),
            # "point_label": torch.tensor(point_labels).float(),
            "point_coords_num": torch.tensor(point_coords_num),
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    

    

if __name__ == '__main__':


    tr_dataset = NpyDataset(data_root, data_aug=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, batch in enumerate(tr_dataloader):
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break