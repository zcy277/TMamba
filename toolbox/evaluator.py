import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

def _get_2dshape(shape, *, zero=True):
    """Helper function to parse shape tuple."""
    if not isinstance(shape, Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1
    assert min(shape) >= minv, f'invalid shape: {shape}'
    return shape

def _pad_image_to_shape(img, shape, value):
    """
    Pads a torch tensor image to a target shape.
    img: [C, H, W] tensor
    """
    shape = _get_2dshape(shape)
    
    c, h, w = img.shape
    
    pad_height = shape[0] - h if shape[0] - h > 0 else 0
    pad_width = shape[1] - w if shape[1] - w > 0 else 0

    # F.pad expects padding in the order of (left, right, top, bottom)
    padding = (
        pad_width // 2,                 # left
        pad_width // 2 + pad_width % 2, # right
        pad_height // 2,                # top
        pad_height // 2 + pad_height % 2  # bottom
    )
    
    # Unsqueeze to add batch dimension for F.pad, then squeeze back
    img_padded = F.pad(img.unsqueeze(0), padding, mode='constant', value=value)
    
    return img_padded.squeeze(0)


class Evaluator(object):
    """
    A dedicated evaluator class that replicates the precise evaluation logic of CFDHI-Net.
    This includes multi-scale, flipping, sliding-window inference, and score aggregation.
    """
    def __init__(self, model, dataset_cfg, num_classes, device):
        self.model = model
        self.device = device
        self.num_classes = num_classes

        # --- 使用更明确的 eval_crop_size，如果不存在则回退到 crop_size ---
        eval_crop_size_str = dataset_cfg.get('eval_crop_size', dataset_cfg.get('crop_size', '480 640'))
        self.crop_size = [int(x) for x in eval_crop_size_str.split()]
        self.stride_rate = dataset_cfg.get('eval_stride_rate', 2/3)
        self.multi_scales = [float(x) for x in dataset_cfg.get('eval_scales', '1.0').split()]
        self.is_flip = dataset_cfg.get('eval_flip', 'false').lower() in ('true', '1', 't')
        
        self.model.eval()

    def process_image(self, image, depth):
        """
        Processes a single image-depth pair using sliding-window inference.
        """
        ori_h, ori_w = image.shape[2], image.shape[3]
        processed_pred = torch.zeros((1, self.num_classes, ori_h, ori_w), device=self.device)

        for scale in self.multi_scales:
            new_h, new_w = int(ori_h * scale), int(ori_w * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True)
            scaled_depth = F.interpolate(depth, size=(new_h, new_w), mode='bilinear', align_corners=True)

            for flip in [False, True] if self.is_flip else [False]:
                inputs_img = torch.flip(scaled_img, dims=[3]) if flip else scaled_img
                inputs_depth = torch.flip(scaled_depth, dims=[3]) if flip else scaled_depth
                
                padded_img = _pad_image_to_shape(inputs_img.squeeze(0), self.crop_size, 0).unsqueeze(0)
                padded_depth = _pad_image_to_shape(inputs_depth.squeeze(0), self.crop_size, 0).unsqueeze(0)
                
                pad_h, pad_w = padded_img.shape[2], padded_img.shape[3]
                
                stride_h = int(np.ceil(self.crop_size[0] * self.stride_rate))
                stride_w = int(np.ceil(self.crop_size[1] * self.stride_rate))
                grid_h = int(np.ceil(float(pad_h - self.crop_size[0]) / stride_h)) + 1
                grid_w = int(np.ceil(float(pad_w - self.crop_size[1]) / stride_w)) + 1

                score_map = torch.zeros((1, self.num_classes, pad_h, pad_w), device=self.device)
                crops_img, crops_depth, coords = [], [], []

                for index_h in range(grid_h):
                    for index_w in range(grid_w):
                        s_h, e_h = index_h * stride_h, min(index_h * stride_h + self.crop_size[0], pad_h)
                        s_h = e_h - self.crop_size[0]
                        s_w, e_w = index_w * stride_w, min(index_w * stride_w + self.crop_size[1], pad_w)
                        s_w = e_w - self.crop_size[1]
                        
                        crops_img.append(padded_img[:, :, s_h:e_h, s_w:e_w])
                        crops_depth.append(padded_depth[:, :, s_h:e_h, s_w:e_w])
                        coords.append((s_h, e_h, s_w, e_w))

                if crops_img:
                    crops_img_batch = torch.cat(crops_img, dim=0)
                    crops_depth_batch = torch.cat(crops_depth, dim=0)
                    
                    with torch.no_grad():
                        score_batch = self.model(crops_img_batch, crops_depth_batch)

                    for i in range(len(coords)):
                        s_h, e_h, s_w, e_w = coords[i]
                        score_map[:, :, s_h:e_h, s_w:e_w] += torch.exp(score_batch[i].unsqueeze(0))

                score = score_map[:, :, :new_h, :new_w]
                if flip:
                    score = torch.flip(score, dims=[3])
                
                score = F.interpolate(score, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
                processed_pred += score
        
        return processed_pred
