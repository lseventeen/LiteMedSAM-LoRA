from typing import Any, Dict, List
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from models.ImageEncoder.tinyvit.tiny_vit import TinyViT
import torch.nn as nn
import torch
import torch.nn.functional as F
class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    # def forward(self, image, boxes=None,masks=None):
    #     image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

    #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #         points=None,
    #         boxes=boxes,
    #         masks=masks,
    #     )
    #     low_res_masks, iou_predictions = self.mask_decoder(
    #         image_embeddings=image_embedding, # (B, 256, 64, 64)
    #         image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
    #         sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
    #         dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
    #         multimask_output=False,
    #       ) # (B, 1, 256, 256)

    #     return low_res_masks, iou_predictions
    



    def forward(self, batched_input: List[Dict[str, Any]],
        multimask_output: bool,):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # input_images = torch.stack(
        #     [self.preprocess(x["image"]) for x in batched_input], dim=0
        # )
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        iou_outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
            iou_outputs.append(iou_predictions)

        outputs = torch.stack([out for out in outputs], dim=0)
        iou_outputs = torch.stack(iou_outputs, dim=0)
        return outputs, iou_outputs

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks



def build_model(args):
    image_encoder = TinyViT(
    args = args,
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    model = MedSAM_Lite(
        image_encoder = image_encoder,
        mask_decoder = mask_decoder,
        prompt_encoder = prompt_encoder
    )
    return model