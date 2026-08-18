import torch
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Config_VSSM():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 3407
    deterministic_setup =  True  #True  False

    cudnn_benchmark = True

    resume_path = None  # e.g., "./logs/Save_VSSM_IRS/basev3/Net_latest.pth"
    continue_path = None # e.g., "./logs/Save_VSSM_IRS/basev3/Net_best.pth"
    batch_size = 4
    lr_start =  3e-4   #6e-5  3e-4
    weight_decay = 1e-4     #1e-2  5e-5  1e-4
    num_epochs = 200
    num_workers = 16
    warm_up_epochs = 10
    lr_power = 0.9    #0.9
    use_warmup_poly_lr_scheduler = True
    model_cfgs = {
        'backbone_name': 'tiny_v1',        # 'tiny' or 'small'  'tiny_v1' or 'small_v1'
        'use_shared_backbone': True,
        'deep_supervision': False,
        'freeze_backbone': True,
        'decoder_kwargs': {},
        'fusion_scan_mode': 'v2',
        'encoder_override_kwargs': {},
        'SE_depth': [1,1,1,1],
    }
    
    # --- Dataset Classes ---
    num_classes = 9      # For IRSEG dataset
    pst_num_classes = 5  # For PST900 dataset
    nyu_num_classes = 40  # For NYU Depth V2 dataset
    sun_num_classes = 37  # For SUN RGB-D dataset

    LOGS_ROOT_DIR = "./logs"
    SAVE_VSSM_IRS_ROOT = os.path.join(LOGS_ROOT_DIR, "Save_Ablation")
    PREDICT_VSSM_IRS_ROOT = os.path.join(LOGS_ROOT_DIR, "Predict_Ablation")
    # SAVE_VSSM_IRS_ROOT = os.path.join(LOGS_ROOT_DIR, "Save_VSSM_IRS")
    # PREDICT_VSSM_IRS_ROOT = os.path.join(LOGS_ROOT_DIR, "Predict_VSSM_IRS")
    SAVE_VSSM_PST_ROOT = os.path.join(LOGS_ROOT_DIR, "Save_VSSM_PST")
    PREDICT_VSSM_PST_ROOT = os.path.join(LOGS_ROOT_DIR, "Predict_VSSM_PST")
    SAVE_VSSM_NYU_ROOT = os.path.join(LOGS_ROOT_DIR, "Save_VSSM_NYU")
    PREDICT_VSSM_NYU_ROOT = os.path.join(LOGS_ROOT_DIR, "Predict_VSSM_NYU")
    SAVE_VSSM_SUN_ROOT = os.path.join(LOGS_ROOT_DIR, "Save_VSSM_SUN")
    PREDICT_VSSM_SUN_ROOT = os.path.join(LOGS_ROOT_DIR, "Predict_VSSM_SUN")

    default_exp_name_irs = "test/baseline"
    default_exp_name_pst = "default_pst_exp"
    default_exp_name_nyu = "default_nyu_exp"
    default_exp_name_sun = "default_sun_exp"

    # --- Dataset-specific Configurations (保持不变) ---
    defult_irs_cfg = {
        "inputs": "rgbd",
        "dataset": "irseg",
        "root": "./Datasets/dataset",
        "n_classes": 9,
        "id_unlabel": -1,
        # "brightness": 0.5,
        # "contrast": 0.5,
        # "saturation": 0.5,
        "p": 0.5,
        "scales_array": [0.5, 0.75, 1, 1.25, 1.5, 1.75], 
        "crop_size": "480 640",
        "eval_crop_size": "480 640", # 评估时的窗口尺寸
        "eval_stride_rate": 2 / 3,
        "eval_scales": "1.0", 
        "eval_flip": "false", 
        "ims_per_gpu": 4,
        "num_workers": 1,
        "lr_start": 6e-5,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_power": 0.9,
        "epochs": 500,
        "loss": "crossentropy",
        "class_weight": "enet"
    }

    defult_pst_cfg = {
        "inputs": "rgbd",
        "dataset": "pst900",
        "root": "./Datasets/PST900_RGBT_Dataset",
        "n_classes": 5,
        "id_unlabel": -1,
        # "brightness": 0.5,
        # "contrast": 0.5,
        # "saturation": 0.5,
        "p": 0.5,
        "scales_array": [0.5, 0.75, 1, 1.25, 1.5, 1.75],
        "crop_size": "720 1280",
        "eval_crop_size": "720 1280", # 评估时的窗口尺寸
        "eval_stride_rate": 2 / 3,
        "eval_scales": "1.0",
        "eval_flip": "false",
        "ims_per_gpu": 4,
        "num_workers": 1,
        "lr_start": 6e-5,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_power": 0.9,
        "epochs": 500,
        "loss": "crossentropy",
        "class_weight": "enet"
    }

    # --- NYU Depth V2 Dataset Configuration ---
    defult_nyu_cfg = {
        "inputs": "rgbd",
        "dataset": "nyu",
        "root": "/home/zcy277/MyProject/RGB-T-SS/Datasets/NYUDepthv2",
        "n_classes": 40,
        "id_unlabel": 255,  # Ignore index for NYU
        "gt_transform": True,  # Labels start from 1, need to subtract 1
        "p": 0.5,
        "scales_array": [0.5, 0.75, 1, 1.25, 1.5, 1.75],
        "crop_size": "480 640",
        "eval_crop_size": "480 640",
        "eval_stride_rate": 2 / 3,
        "eval_scales": "0.75 1.0 1.25",
        "eval_flip": "true",
        "ims_per_gpu": 4,
        "num_workers": 1,
        "lr_start": 6e-5,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_power": 0.9,
        "epochs": 500,
        "loss": "crossentropy",
        "class_weight": "enet"
    }

    # --- SUN RGB-D Dataset Configuration ---
    defult_sun_cfg = {
        "inputs": "rgbd",
        "dataset": "sun",
        "root": "/home/zcy277/MyProject/RGB-T-SS/Datasets/SUNRGBD",
        "n_classes": 37,
        "id_unlabel": 255,  # Ignore index for SUN RGB-D
        "gt_transform": True,  # Labels start from 1, need to subtract 1
        "p": 0.5,
        "scales_array": [0.5, 0.75, 1, 1.25, 1.5, 1.75],
        "crop_size": "480 640",
        "eval_crop_size": "480 640",
        "eval_stride_rate": 2 / 3,
        "eval_scales": "0.75 1.0 1.25",
        "eval_flip": "true",
        "ims_per_gpu": 4,
        "num_workers": 1,
        "lr_start": 6e-5,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_power": 0.9,
        "epochs": 500,
        "loss": "crossentropy",
        "class_weight": "enet"
    }
