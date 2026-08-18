"""Train the final TMamba model on IRS or PST900."""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from config import Config_VSSM
from Models.TMamba import TMamba
from toolbox import runningScore
from toolbox.datasets.irseg import IRSeg
from toolbox.datasets.pst900 import PSTSeg
from toolbox.scheduler.lr_scheduler import WarmupPolyLRScheduler


def setup_seed(seed, deterministic_setup=False, cudnn_benchmark=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_setup:
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = cudnn_benchmark


def build_datasets(cfg, dataset_name, experiment_name):
    if dataset_name == "IRS":
        dataset_cfg = cfg.defult_irs_cfg
        num_classes = cfg.num_classes
        train_dataset = IRSeg(
            cfg=dataset_cfg,
            root=dataset_cfg["root"],
            mode="trainval",
        )
        evaluation_dataset = IRSeg(
            cfg=dataset_cfg,
            root=dataset_cfg["root"],
            mode="test",
        )
        experiment_name = experiment_name or cfg.default_exp_name_irs
        checkpoint_dir = os.path.join(cfg.SAVE_VSSM_IRS_ROOT, experiment_name)
        result_dir = os.path.join(cfg.PREDICT_VSSM_IRS_ROOT, experiment_name)
    else:
        dataset_cfg = cfg.defult_pst_cfg
        num_classes = cfg.pst_num_classes
        train_dataset = PSTSeg(
            cfg=dataset_cfg,
            root=dataset_cfg["root"],
            mode="train",
        )
        evaluation_dataset = PSTSeg(
            cfg=dataset_cfg,
            root=dataset_cfg["root"],
            mode="test",
        )
        experiment_name = experiment_name or cfg.default_exp_name_pst
        checkpoint_dir = os.path.join(cfg.SAVE_VSSM_PST_ROOT, experiment_name)
        result_dir = os.path.join(cfg.PREDICT_VSSM_PST_ROOT, experiment_name)

    ignore_index = dataset_cfg.get("id_unlabel", -1)
    return (
        train_dataset,
        evaluation_dataset,
        num_classes,
        ignore_index,
        experiment_name,
        checkpoint_dir,
        result_dir,
    )


def optimizer_parameters(model, learning_rate, weight_decay):
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() == 1 or name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return [
        {
            "params": decay_parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "lr": learning_rate,
            "weight_decay": 0.0,
        },
    ]


def append_log(log_path, message):
    print(message)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


@torch.no_grad()
def evaluate(loader, model, metrics, device):
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for sample in loader:
        image = sample["image"].to(device)
        depth = sample["depth"].to(device)
        label = sample["label"].to(device)

        logits, loss = model(image, depth, label)
        prediction = logits.argmax(dim=1).cpu().numpy()
        metrics.update(label.cpu().numpy(), prediction)
        total_loss += loss.item()

    mean_loss = total_loss / len(loader)
    mean_iou = metrics.get_scores()[0]["mIou: "]
    return mean_loss, mean_iou


def train(args):
    cfg = Config_VSSM()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr_start = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.workers is not None:
        cfg.num_workers = args.workers

    setup_seed(cfg.seed, cfg.deterministic_setup, cfg.cudnn_benchmark)
    (
        train_dataset,
        evaluation_dataset,
        num_classes,
        ignore_index,
        experiment_name,
        checkpoint_dir,
        result_dir,
    ) = build_datasets(cfg, args.dataset, args.exp_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "training_log.txt")

    pin_memory = cfg.device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = TMamba(
        num_classes=num_classes,
        ignore_index=ignore_index,
    ).to(cfg.device)
    optimizer = optim.AdamW(
        optimizer_parameters(model, cfg.lr_start, cfg.weight_decay),
        betas=(0.9, 0.999),
        lr=cfg.lr_start,
    )

    total_iterations = cfg.num_epochs * len(train_loader)
    if cfg.use_warmup_poly_lr_scheduler:
        scheduler = WarmupPolyLRScheduler(
            optimizer,
            T_max=total_iterations,
            warmup_iters=cfg.warm_up_epochs * len(train_loader),
            power=cfg.lr_power,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iterations,
            eta_min=1e-6,
        )

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"Dataset: {args.dataset}\n"
            f"Experiment: {experiment_name}\n"
            f"Classes: {num_classes}\n"
            f"Epochs: {cfg.num_epochs}\n"
            f"Batch size: {cfg.batch_size}\n"
            f"Learning rate: {cfg.lr_start}\n"
            f"Weight decay: {cfg.weight_decay}\n"
        )

    metrics = runningScore(num_classes, ignore_index=ignore_index)
    best_iou = 0.0
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for step, sample in enumerate(train_loader, start=1):
            image = sample["image"].to(cfg.device)
            depth = sample["depth"].to(cfg.device)
            label = sample["label"].to(cfg.device)

            optimizer.zero_grad()
            _, loss = model(image, depth, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{cfg.num_epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = total_loss / len(train_loader)
        evaluation_loss, mean_iou = evaluate(
            evaluation_loader,
            model,
            metrics,
            cfg.device,
        )
        duration = time.time() - epoch_start

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "Net_best.pth"),
            )

        append_log(
            log_path,
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"eval_loss={evaluation_loss:.4f}, mIoU={mean_iou:.4f}, "
            f"duration={duration:.2f}s, best_mIoU={best_iou:.4f} "
            f"(epoch {best_epoch})",
        )

    append_log(
        log_path,
        f"Training finished. Best mIoU={best_iou:.4f} at epoch {best_epoch}.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TMamba for RGB-thermal semantic segmentation."
    )
    parser.add_argument("--dataset", required=True, choices=["IRS", "PST"])
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
