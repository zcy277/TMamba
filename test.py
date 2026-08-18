"""Evaluate the final TMamba model on IRS or PST900."""

import argparse
import os
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config_VSSM
from Models.TMamba import TMamba
from toolbox.datasets.irseg import IRSeg
from toolbox.datasets.pst900 import PSTSeg
from toolbox.evaluator import Evaluator
from toolbox.metrics import averageMeter, runningScore
from toolbox.utils import class_to_RGB


def evaluation_setup(cfg, dataset_name, experiment_name):
    if dataset_name == "IRS":
        dataset_cfg = cfg.defult_irs_cfg
        num_classes = cfg.num_classes
        experiment_name = experiment_name or cfg.default_exp_name_irs
        checkpoint_dir = os.path.join(cfg.SAVE_VSSM_IRS_ROOT, experiment_name)
        result_dir = os.path.join(cfg.PREDICT_VSSM_IRS_ROOT, experiment_name)
        default_modes = ["test", "test_day", "test_night"]
    else:
        dataset_cfg = cfg.defult_pst_cfg
        num_classes = cfg.pst_num_classes
        experiment_name = experiment_name or cfg.default_exp_name_pst
        checkpoint_dir = os.path.join(cfg.SAVE_VSSM_PST_ROOT, experiment_name)
        result_dir = os.path.join(cfg.PREDICT_VSSM_PST_ROOT, experiment_name)
        default_modes = ["test"]

    return (
        dataset_cfg,
        num_classes,
        experiment_name,
        checkpoint_dir,
        result_dir,
        default_modes,
    )


def build_dataset(dataset_name, dataset_cfg, mode):
    if dataset_name == "IRS":
        dataset = IRSeg(
            cfg=dataset_cfg,
            root=dataset_cfg["root"],
            mode=mode,
        )
        color_map = dataset.cmap
    else:
        dataset = PSTSeg(
            cfg=dataset_cfg,
            root=dataset_cfg["root"],
            mode=mode,
        )
        color_map = dataset.tcsvt2023_DSGBINet_cmap
    return dataset, color_map


def write_metrics(log_file, mode, time_meter, metrics):
    with open(log_file, "a", encoding="utf-8") as output:
        output.write(f"\nMode: {mode}\n")
        output.write(f"Average inference time: {time_meter.avg:.4f}s\n")
        for name, value in metrics[0].items():
            output.write(f"{name}{value:.4f}\n")
        output.write("IoU per class:\n")
        for class_index, value in metrics[1].items():
            output.write(f"  Class {class_index}: {value:.4f}\n")
        output.write("Accuracy per class:\n")
        for class_index, value in metrics[2].items():
            output.write(f"  Class {class_index}: {value:.4f}\n")


def evaluate(args):
    cfg = Config_VSSM()
    (
        dataset_cfg,
        num_classes,
        experiment_name,
        checkpoint_dir,
        result_dir,
        default_modes,
    ) = evaluation_setup(cfg, args.dataset, args.exp_name)

    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = TMamba(
        num_classes=num_classes,
        ignore_index=dataset_cfg.get("id_unlabel", -1),
        load_pretrained_backbone=False,
    ).to(cfg.device)
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, "evaluation_results.txt")
    with open(log_file, "w", encoding="utf-8") as output:
        output.write(
            f"Dataset: {args.dataset}\n"
            f"Experiment: {experiment_name}\n"
            f"Checkpoint: {args.checkpoint_name}\n"
        )

    evaluator = Evaluator(
        model,
        dataset_cfg,
        num_classes,
        cfg.device,
    )
    modes = args.modes or default_modes

    for mode in modes:
        dataset, color_map = build_dataset(args.dataset, dataset_cfg, mode)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
        prediction_dir = os.path.join(result_dir, mode)
        if args.save_predict:
            os.makedirs(prediction_dir, exist_ok=True)

        metrics = runningScore(
            num_classes,
            ignore_index=dataset_cfg.get("id_unlabel", -1),
        )
        time_meter = averageMeter()

        with torch.no_grad():
            for sample in tqdm(loader, desc=mode):
                start_time = time.perf_counter()
                image = sample["image"].to(cfg.device)
                depth = sample["depth"].to(cfg.device)
                label = sample["label"].to(cfg.device)

                score = evaluator.process_image(image, depth)
                prediction = score.argmax(dim=1).cpu().numpy()
                metrics.update(label.cpu().numpy(), prediction)
                time_meter.update(
                    time.perf_counter() - start_time,
                    n=image.size(0),
                )

                if args.save_predict:
                    prediction_rgb = class_to_RGB(
                        prediction[0],
                        N=len(color_map),
                        cmap=color_map,
                    )
                    Image.fromarray(prediction_rgb).save(
                        os.path.join(prediction_dir, sample["label_path"][0])
                    )

        scores = metrics.get_scores()
        write_metrics(log_file, mode, time_meter, scores)
        print(
            f"{mode}: mIoU={scores[0]['mIou: ']:.4f}, "
            f"average_time={time_meter.avg:.4f}s"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TMamba for RGB-thermal semantic segmentation."
    )
    parser.add_argument("--dataset", required=True, choices=["IRS", "PST"])
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--checkpoint_name", default="Net_best.pth")
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--save_predict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
