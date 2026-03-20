import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from rosaid.core.defs import DNP3_CLASSES, IEC104_CLASSES, ROSIDS23_CLASSES
from rosaid.data.dataset import (
    DFDataSet,
    NormalImageType,
    SessionImageDataConfig,
    TorchImageDataset,
)
from rosaid.models.cnn import BlockCNN2D
from rosaid.models.image_model import ImageClfModel
from rosaid.trainers.trainer import NNTrainer, NNTrainerConfig

# argument parser to accept: image_type[normalized,normal], backbone, max_data
parser = argparse.ArgumentParser(description="Session Image Trainer Configuration")
parser.add_argument(
    "--image_type",
    type=str,
    choices=["normal", "normalized"],
    default="normal",
    help="Type of images to use for training. 'normal' for raw images, 'normalized' for normalized images.",
)
parser.add_argument(
    "--backbone",
    type=str,
    # default="mobilenet_v3_large",
    default="blockcnn2d",
    choices=[
        "mobilenet_v3_large",
        "resnet18",
        "blockcnn2d",
    ],
    help="Backbone model to use for image classification.",
)
parser.add_argument(
    "--max_data",
    type=int,
    default=100,
    help="Maximum number of data points to use. Use -ve for all data.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch Size.",
)
parser.add_argument(
    "--sampling_method",
    type=str,
    choices=["nosampling", "oversampling", "undersampling"],
    default="nosampling",
    help="Sampling method to use for the dataset.",
)
parser.add_argument(
    "--num_samples_per_epoch",
    type=int,
    default=10000,
    help="Number of samples to use per epoch. Default is 10000.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    # default=r"rosaid\data\120_timeout_dnp3_sessions",
    default=r"rosaid\data\iec104_sessions",
)
parser.add_argument(
    "--data_type",
    type=str,
    choices=["DNP3", "IEC104", "ROSIDS23"],
    default="IEC104",
    help="Type of data to use: 'DNP3' or 'IEC104'.",
)
parser.add_argument(
    "--project_dir",
    type=str,
    default=r"rosaid",
    help="Project directory where results will be saved.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1000,
    help="Number of epochs to train the model.",
)
parser.add_argument(
    "--clf_mode",
    type=str,
    choices=["binary", "multiclass"],
    default="binary",
    help="Classification mode: 'binary' for binary classification, 'multiclass' for multiclass classification.",
)
parser.add_argument(
    "--attack_only",
    action="store_true",
    default=False,
    help="Use only attack samples for training.",
)
parser.add_argument(
    "--normal_image_type",
    type=str,
    choices=NormalImageType._member_names_,
    default=NormalImageType._member_names_[0],
    help="Type of normal image to use.",
)

args = parser.parse_args()
batch_size = args.batch_size
use_normalized = args.image_type.lower() == "normalized"
normalized_str = "normalized_frequency" if use_normalized else ""
sampling_method = args.sampling_method
data_dir = Path(args.data_dir)
project_dir = Path(args.project_dir)
data_type = args.data_type

normal_image_type = NormalImageType[args.normal_image_type]

if data_type == "DNP3":
    labels = DNP3_CLASSES
    # max is 138
    num_pkts = 138
elif data_type == "IEC104":
    labels = IEC104_CLASSES
    # 99 pctl is 289 so have some buffer
    num_pkts = 290
elif data_type == "ROSIDS23":
    labels = ROSIDS23_CLASSES
    # 446 is max, 78 is 99 pctl bt 406 is max of normal
    # but those seem to be outliers and 100 could be sufficient
    num_pkts = 100
else:
    raise ValueError(f"Unsupported data type: {data_type}")
# bcz of fixed img types
# num_pkts = 256
logger.info(f"Args: {args}")

combine_attacks = True if args.clf_mode == "binary" else False
attack_only = args.attack_only


expt_name = "image_classification"
normal_label = labels[0]
num_cpu = max(8, os.cpu_count() // 3)
num_cpu = min(num_cpu, os.cpu_count() - 2)
logger.info(f"Using {num_cpu} CPU cores for data loading.")

if __name__ == "__main__":
    # Configuration parameters
    config = SessionImageDataConfig(
        max_data=args.max_data,
        session_images_dir=data_dir / "session_images",
        labels_file=data_dir / "labelled_sessions.csv",
        sampling_method=sampling_method.lower(),
        use_normalized=use_normalized,
        combine_attacks=combine_attacks,
        attack_only=attack_only,
        labels=labels,
        normal_label=normal_label,
        num_pkts=num_pkts,
        balanced_batches=True,
        normal_image_type=normal_image_type,
    )

    # Load the dataset
    train_ds, test_ds = DFDataSet(config=config).load_data()
    img, lbl, lbl_str = test_ds[0]

    if args.image_type != 'normalized':
        normalized_str += f"_{normal_image_type.name}"

    expt_dir = project_dir / "results" / expt_name /f"{data_type}{normalized_str}"
    if not expt_dir.exists():
        expt_dir.mkdir(parents=True)
    today_date = datetime.now().date().strftime("%Y%m%d")

    run_name = f"{args.backbone}_{sampling_method}_{args.clf_mode}_{today_date}"
    run_dir = expt_dir / run_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    num_classes = len(train_ds.label_encoding)
    if args.clf_mode == "binary":
        num_classes = 1
    ch = 1
    if img.ndim == 3:
        ch = 3
    # Initialize the model
    if args.backbone == "blockcnn2d":
        model = BlockCNN2D(
            input_size=(ch, img.shape[0], img.shape[1]),
            output_size=num_classes,
            global_pool="avg",
        )
    else:
        model = ImageClfModel(
            in_channel=ch,
            num_classes=num_classes,
            backbone=args.backbone,
        )
    try:
        img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(run_dir / f"{lbl_str}.png"), img)
    except Exception as e:
        logger.error(f"Error saving image: {e}")
    # Initialize the trainer
    trainer = NNTrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name=expt_name,
            run_name=run_name,
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=25,
            log_mlflow=False,
            weight_decay=1e-5,
            # optimizer="adamw",
            # best_model_metric="f1_score",
            # best_model_metric_greater=True,
            is_binary_classification=num_classes == 1,
            number_of_workers=num_cpu,
        ),
        model=model,
        train_dataset=TorchImageDataset(train_ds),
        val_dataset=TorchImageDataset(test_ds),
    )
    # Train the model
    trainer.train()
    # Plot the training metrics
    trainer.plot_metrics()
    logger.stop()
