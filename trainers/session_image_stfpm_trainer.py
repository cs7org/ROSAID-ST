import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from rosaid.core.defs import DNP3_CLASSES, IEC104_CLASSES
from rosaid.data.dataset import DFDataSet, SessionImageDataConfig, TorchImageDataset
from rosaid.data.dynamic_dataset import (
    DynamicSessionDFDataset,
    DynamicTorchSessionImageDataset,
)
from rosaid.models.cnn import BlockCNN2D
from rosaid.models.packet_flow_model import (
    DynamicPacketModel,
    PacketFlowClassifier,
    PacketModel,
)
from rosaid.core.defs import ROSIDS23_CLASSES

from rosaid.models.stfpm import STFPModel as STFPModelCustom
from rosaid.trainers.trainer_stfpm_image import STFPMTrainer
from rosaid.trainers.trainer_stfpm_image import STFPMTrainerConfig as NNTrainerConfig

parser = argparse.ArgumentParser(description="Session Image Trainer Configuration")
parser.add_argument(
    "--image_type",
    type=str,
    choices=["normal", "normalized"],
    default="normal",
    help="Type of images to use for training. 'normal' for raw images, 'normalized' for normalized images.",
)
parser.add_argument(
    "--data_type",
    type=str,
    choices=["DNP3", "IEC104", "ROSIDS23"],
    default="IEC104",
    help="Type of data to use for training.",
)
parser.add_argument(
    "--backbone",
    type=str,
    # default="resnet18",
    default="blockcnn2d",
    help="Backbone model to use for image classification.",
)
parser.add_argument(
    "--teacher_model",
    type=str,
    default=r"rosaid\results\image_classification\IEC104_blockcnn2d__nosampling_multiclass_20251229\best_model_full.pth",
    help="Path to the pre-trained teacher model.",
)
parser.add_argument(
    "--max_data",
    type=int,
    default=50,
    help="Maximum number of data points to use. Use -ve for all data.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
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
    default=r"rosaid\data\iec104_sessions",
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
    default="multiclass",
    help="Classification mode: 'binary' for binary classification, 'multiclass' for multiclass classification.",
)
parser.add_argument(
    "--attack_only",
    action="store_true",
    default=False,
    help="Use only attack samples for training.",
)

args = parser.parse_args()
batch_size = args.batch_size
use_normalized = args.image_type.lower() == "normalized"
normalized_str = "normalized" if use_normalized else "_"
sampling_method = args.sampling_method
data_dir = Path(args.data_dir)
project_dir = Path(args.project_dir)
data_type = args.data_type

logger.info(f"Args: {args}")


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


backbone = args.backbone
if "blockcnn2d" in args.backbone.lower():
    backbone = "blockcnn2d"

expt_name = "image_stfpm"
normal_label = labels[0]
if __name__ == "__main__":
    today_date = datetime.datetime.now().date().strftime("%Y%m%d")
    teacher_pth = Path(args.teacher_model)
    tchr = teacher_pth.parent.name.split("_")[0]
    run_name = f"{tchr}_{args.backbone}_{normalized_str}_{data_type}_{sampling_method}_{today_date}"

    teacher_model = torch.load(teacher_pth, weights_only=False).eval()
    if isinstance(teacher_model, BlockCNN2D):
        # Configuration parameters
        config = SessionImageDataConfig(
            max_data=args.max_data,
            session_images_dir=data_dir / "session_images",
            labels_file=data_dir / "labelled_sessions.csv",
            sampling_method=sampling_method.lower(),
            use_normalized=use_normalized,
            labels=labels,
            normal_label=normal_label,
            num_pkts=num_pkts,
            balanced_batches=False,
        )

        # Load the dataset
        train_ds, test_ds = DFDataSet(config=config).load_data()
        student_model = BlockCNN2D(
            input_size=teacher_model.input_size,
            hidden_channels=teacher_model.hidden_channels,
            output_size=teacher_model.output_size,
            dropout_rate=teacher_model.dropout_rate,
            use_batchnorm=teacher_model.use_batchnorm,
            kernel_size=teacher_model.kernel_size,
            flatten_features=teacher_model.flatten_features,
        )

        model = STFPModelCustom(
            layers=[
                "block_0",
                "block_1",
                "block_2",
                "block_3",
                # "block_4",
                # "block_5",
                # "block_6",
            ],
            student=student_model,
            teacher=teacher_model,
        )
    elif isinstance(teacher_model, PacketFlowClassifier):
        teacher_model = teacher_model.pfe
        channels = teacher_model.channels
        global_pool_type = teacher_model.global_pool_type
        feature_dim = teacher_model.feature_dim

        config = SessionImageDataConfig(
            max_data=args.max_data,
            session_images_dir=data_dir / "session_images",
            labels_file=data_dir / "labelled_sessions.csv",
            sampling_method=sampling_method.lower(),
            labels=labels,
            normal_label=normal_label,
            num_pkts=num_pkts,
            balanced_batches=False,
        )

        # Load the dataset
        train_ds, test_ds = DynamicSessionDFDataset(config=config).load_data()
        TorchImageDataset = DynamicTorchSessionImageDataset

        if isinstance(teacher_model, DynamicPacketModel):
            student_model = DynamicPacketModel(
                channels=channels,
                global_pool_type=global_pool_type,
                feature_dim=feature_dim,
            )
        else:
            student_model = PacketModel(
                input_size=teacher_model.input_size,
                channels=channels,
                global_pool_type=global_pool_type,
                feature_dim=feature_dim,
            )

        model = STFPModelCustom(
            layers=[],  # dont need layers
            student=student_model,
            teacher=teacher_model,
        )
    else:
        from rosaid.models.stfpm_anm import STFPMModel

        # # Configuration parameters
        config = SessionImageDataConfig(
            max_data=args.max_data,
            session_images_dir=data_dir / "session_images",
            labels_file=data_dir / "labelled_sessions.csv",
            sampling_method=sampling_method.lower(),
            use_normalized=use_normalized,
            labels=labels,
            normal_label=normal_label,
            num_pkts=num_pkts,
            balanced_batches=False,
        )

        # Load the dataset
        train_ds, test_ds = DFDataSet(config=config).load_data()
        model = STFPMModel(
            layers=["layer1", "layer2", "layer3"],
            backbone=args.backbone,
            teacher_model=args.teacher_model,
        )
        # raise ValueError("Unsupported teacher model type.")

    logger.info(f"Train dataset size: {len(train_ds)}")
    # NOTE: train only on normal samples
    train_ds.data_df = train_ds.data_df.query(f'label=="{normal_label}"')
    train_ds.label_encoding = {normal_label: train_ds.label_encoding[normal_label]}
    logger.info(f"Filtered Train dataset size: {len(train_ds)}")
    img, lbl, lbl_str = test_ds[0]

    expt_dir = project_dir / "results" / expt_name
    if not expt_dir.exists():
        expt_dir.mkdir(parents=True)

    cv2.imwrite(str(expt_dir / f"{run_name}.png"), (img * 255).astype(np.uint8))
    # Initialize the trainer
    trainer = STFPMTrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name=expt_name,
            run_name=run_name,
            epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=25,
            log_mlflow=False,
            weight_decay=1e-5,
            optimizer="adamw",
            metrics=[],
        ),
        model=model,
        train_dataset=TorchImageDataset(train_ds),
        val_dataset=TorchImageDataset(test_ds),
    )
    # load existing model if exists
    # trainer.model = torch.load(
    #     r"rosaid\results\stfpm\resnet18_nosampling\last_model_full.pth",
    #     weights_only=False,
    #     map_location=trainer.device,
    # )
    # trainer.optimizer.load_state_dict(
    #     torch.load(
    #         r"rosaid\results\stfpm\resnet18_nosampling\optimizer_state.pth",
    #         map_location=trainer.device,
    #     )
    # )
    # Train the model
    trainer.train()
    # Plot the training metrics
    trainer.plot_metrics()
    logger.stop()
