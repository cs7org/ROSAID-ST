from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from rosaid.core.defs import DataType, NormalImageType, SamplingMethod
from rosaid.utils.image import get_normal_image


class SessionImageDataConfig(BaseModel):
    session_images_dir: Path = Field(
        default=Path(
            r"E:\MSc Works\IDS\notebooks\120_timeout_dnp3_sessions\session_images"
        ),
        description="Directory to store session images.",
    )
    labels_file: Path = Field(
        default=Path(
            r"E:\MSc Works\IDS\notebooks\120_timeout_dnp3_sessions\labelled_sessions.csv"
        ),
        description="File containing labels for the sessions.",
    )
    train_ratio: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for training.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    labels: list[str] = Field(
        default=[],
        description="List of labels to be used for classification.",
    )
    normal_label: str = Field(
        default="NORMAL",
        description="Label for normal data samples.",
    )
    combine_attacks: bool = Field(
        default=False,
        description="Whether to combine all attack types into a single label.",
    )
    max_data: int | float = Field(
        default=-1,
        description="Maximum number of samples to be used from each class.",
    )
    byte_length: int = Field(
        default=256,  # this is to match normalized images
        description="Length of byte sequences to be used for image generation.",
    )
    num_pkts: int = Field(
        default=138,  # See the summary of num pkts for all. this is max of max
        description="Number of packets to consider for each sample.",
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.NONE,
        description="Method to use for sampling data.",
    )
    use_normalized: bool = Field(
        default=False,
        description="Whether to normalize image pixel values.",
    )
    min_num_pkts: int = Field(
        default=1,
        ge=1,
        description="Minimum number of packets to consider for each sample.",
    )
    label_column: str = Field(
        default="flow_label",
        description="Column name in the labels file that contains the labels.",
    )
    data_labels: list[str] = Field(
        default=[],
        description="List of labels to use for training. If empty, all labels will be used.",
    )
    attack_only: bool = Field(
        default=False,
        description="Whether to use only attack samples for training.",
    )
    balanced_batches: bool = Field(
        default=True,
        description="Whether to create balanced batches during training.",
    )
    normal_image_type: NormalImageType | int = Field(
        default=NormalImageType.ORIGINAL,
        description="Type of normal image to use for training.",
    )
    filter_first_nonzero_columns: bool = Field(
        default=False,
        description="Whether to filter the first non-zero column in the image.",
    )


def image_normalize(image):
    """Normalize image pixel values to [0, 1] range."""
    return image / image.max()


class DFDataSet:
    def __init__(self, config: SessionImageDataConfig):
        self.config = config
        self.train_ratio = config.train_ratio
        self.data_df = None
        self.data_type = None
        self.label_encoding = None
        self.scaler = image_normalize
        self.label_row_ids = defaultdict(list)
        self.random_state = np.random.RandomState(self.config.random_seed)
        self.label_batch_counters = defaultdict(int)

    def load_data(self):
        data_df = pd.read_csv(self.config.labels_file)
        data_df = data_df.query(f"total_matched_pkts>={self.config.min_num_pkts}")
        min_num_pkts = data_df["total_matched_pkts"].min()
        max_num_pkts = data_df["total_matched_pkts"].max()
        min_byte_length = data_df["raw_bytes_min_length"].min()
        max_byte_length = data_df["raw_bytes_max_length"].max()

        logger.info(
            f"Dataset stats - min_num_pkts: {min_num_pkts}, max_num_pkts: {max_num_pkts}, min_byte_length: {min_byte_length}, max_byte_length: {max_byte_length}"
        )
        # shuffle it first
        # data_df = data_df.sample(frac=1, random_state=self.config.random_seed).reset_index(
        #     drop=True
        # )
        self.data_df = data_df.copy()
        if self.config.attack_only:
            logger.info("Filtering dataset to only include attack samples")
            # use self.config.label_column to filter attack samples
            self.data_df = self.data_df.query(
                f"{self.config.label_column} != '{self.config.normal_label}'"
            )
        else:
            logger.info("Using all samples including normal and attack samples")
        self.data_df["label"] = (
            self.data_df[self.config.label_column].astype(str).str.upper()
        )
        # it does not have file_path col
        if self.config.use_normalized:
            logger.info("Using normalized images for training")
            self.data_df["file_path"] = self.data_df.apply(
                lambda row: self.config.session_images_dir
                / f"{row.session_file_name.replace('.pcap', '_normalized.png')}",
                axis=1,
            )
        else:
            self.data_df["file_path"] = self.data_df.apply(
                lambda row: self.config.session_images_dir
                / f"{row.session_file_name.replace('.pcap', '.png')}",
                axis=1,
            )

        if self.config.combine_attacks:
            self.data_df["label"] = self.data_df["label"].apply(
                lambda x: (
                    self.config.normal_label
                    if x == self.config.normal_label
                    else "ATTACK"
                )
            )
            # update label column to combine all attacks into a single label
            self.data_df[self.config.label_column] = self.data_df["label"]
            logger.info("Combined all attack types into a single label")
        logger.info(f"Data loaded into DataFrame with {len(self.data_df)} entries")
        logger.info(f"Label distribution:\n{self.data_df['label'].value_counts()}")

        if self.config.max_data > 0:
            logger.info(f"Limiting dataset to {self.config.max_data} samples per class")
            lbl_cnts = self.data_df["label"].value_counts().to_dict()
            # if max data is ratio then find max of each label
            if isinstance(self.config.max_data, float) and self.config.max_data < 1.0:
                nlbl_cnts = {
                    k: int(v * self.config.max_data) for k, v in lbl_cnts.items()
                }
            else:
                nlbl_cnts = {
                    k: min(self.config.max_data, lbl_cnts[k]) for k in lbl_cnts.keys()
                }
            logger.info(f"Limiting data selection to: {nlbl_cnts}")
            self.data_df = (
                self.data_df.groupby("label", group_keys=False)
                .apply(
                    lambda x: x.sample(
                        nlbl_cnts[x.name],
                        random_state=self.config.random_seed,
                        replace=False,
                    )
                )
                .reset_index(drop=True)
            )
        logger.info(
            f"Final dataset size: {len(self.data_df)} entries after applying max_data limit"
        )
        labels = self.data_df["label"].unique().tolist()
        logger.info(f"Found labels: {labels}")
        if not labels:
            raise ValueError("No labels found in the dataset. Please check the data.")

        # if self.config.data_labels:
        #     logger.info(
        #         f"Filtering dataset to only include specified labels: {self.config.data_labels}"
        #     )
        #     labels = self.config.data_labels
        # else:
        #     labels.sort()
        self.label_encoding = {label: idx for idx, label in enumerate(labels)}
        for label in self.label_encoding.keys():
            lbl = [0] * len(self.label_encoding)
            idx = self.label_encoding[label]
            lbl[idx] = 1
            self.label_encoding[label] = lbl

        train_df, test_df = train_test_split(
            self.data_df,
            train_size=self.train_ratio,
            stratify=self.data_df["label"],
            random_state=self.config.random_seed,
        )
        min_labels = train_df["label"].value_counts().min()
        max_labels = train_df["label"].value_counts().max()
        if self.config.sampling_method == SamplingMethod.UNDERSAMPLE:
            logger.info(f"Undersampling to {min_labels} samples per class for training")
            train_df = (
                train_df.groupby("label")
                .apply(
                    lambda x: x.sample(min_labels, random_state=self.config.random_seed)
                )
                .reset_index(drop=True)
            )
        elif self.config.sampling_method == SamplingMethod.OVERSAMPLE:
            logger.info(f"Oversampling to {max_labels} samples per class for training")
            new_df = train_df.query(f"label == '{self.config.normal_label}'")
            for label in train_df["label"].unique():
                if label == self.config.normal_label:
                    continue
                label_df = train_df.query(f"label == '{label}'")
                if len(label_df) < max_labels:
                    num_samples = max_labels
                    sampled_df = label_df.sample(
                        num_samples, replace=True, random_state=self.config.random_seed
                    )
                    new_df = pd.concat([new_df, sampled_df], ignore_index=True)
            train_df = new_df
        elif self.config.sampling_method == SamplingMethod.NONE:
            logger.info("No sampling applied, using original training data")
        else:
            raise ValueError(
                f"Unsupported sampling method: {self.config.sampling_method}"
            )
        train_df = train_df.sample(
            frac=1, random_state=self.config.random_seed
        ).reset_index(drop=True)
        logger.info(
            f"After sampling, training data label distribution:\n{train_df['label'].value_counts()}"
        )

        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        logger.info(
            f"Split data into {len(self.train_df)} training and {len(self.test_df)} testing samples"
        )
        self.config.labels = labels
        train_dataset, test_dataset = self.get_datasets()
        logger.info(f"Created datasets: {train_dataset} and {test_dataset}")
        logger.info(
            f"{train_dataset.data_type}: {train_dataset.data_df.label.value_counts()}"
        )

        return train_dataset, test_dataset

    def get_datasets(self):
        train_dataset = DFDataSet(self.config.copy())
        train_dataset.data_df = self.train_df
        train_dataset.data_type = DataType.TRAIN
        train_dataset.label_encoding = self.label_encoding

        test_dataset = DFDataSet(self.config.copy())
        test_dataset.data_df = self.test_df
        test_dataset.data_type = DataType.VALIDATION
        test_dataset.label_encoding = self.label_encoding

        logger.info("Mapping label to row indices for training dataset")

        for idx, row in train_dataset.data_df.iterrows():
            train_dataset.label_row_ids[row["label"]].append(idx)

        for idx, row in test_dataset.data_df.iterrows():
            test_dataset.label_row_ids[row["label"]].append(idx)

        # log count for both datasets
        logger.info(
            f"Training dataset label distribution:\n{train_dataset.data_df['label'].value_counts()}"
        )
        logger.info(
            f"Testing dataset label distribution:\n{test_dataset.data_df['label'].value_counts()}"
        )
        return train_dataset, test_dataset

    def __len__(self):
        if self.data_df is not None:
            return len(self.data_df)

    def get_class_row(self, label: str):
        
        if label in self.label_row_ids:
            # randomly select one row id
            row_id = self.random_state.choice(self.label_row_ids[label])
            return self.data_df.iloc[row_id]
        raise ValueError("DataFrame is not initialized.")

    def __getitem__(self, idx):
        if self.data_df is not None:
            if self.config.balanced_batches and (self.data_type == DataType.TRAIN):
                # get label for this idx in uniform random way
                labels = list(self.label_encoding.keys())

                label = self.random_state.choice(labels)
                row = self.get_class_row(label)

            else:
                row = self.data_df.iloc[idx]
        image_path = Path(row["file_path"])
        # if not image_path.exists():
        #     # check if it is in session_images_dir
        #     image_path = image_path.parent / 'session_images' / image_path.name

        self.curr_image_path = image_path
        label = row["label"]
        lbl_string = row.label

        self.label_batch_counters[lbl_string] += 1
        # if 10000 items have been processed log the counters
        counter_sum = sum(self.label_batch_counters.values())
        if idx==len(self.data_df)-1 and self.data_type == DataType.TRAIN:
            logger.info(f"Label batch counters: {dict(self.label_batch_counters)}")

        if image_path.exists():
            img = cv2.imread(str(image_path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # gives 256,256,[3] or 16,16,3
            non_fixed_len_types = [
                NormalImageType.ORIGINAL,
                NormalImageType.FILTERED,
                NormalImageType.NORMALIZED,
                NormalImageType.FILTERED_GRAM,
                NormalImageType.UNFILTERED_GRAM,
                NormalImageType.UNFILTERED_GRAM3D,
                NormalImageType.FILTERED_GRAM3D,
            ]
            if (
                not self.config.use_normalized
                and self.config.normal_image_type not in non_fixed_len_types
            ):
                gray_img = get_normal_image(gray_img, 
                                            self.config.normal_image_type, 
                                            self.config.filter_first_nonzero_columns)

            else:
                if (
                    not self.config.use_normalized
                    and self.config.normal_image_type in non_fixed_len_types
                ):
                    gray_img = get_normal_image(gray_img, self.config.normal_image_type,
                                                self.config.filter_first_nonzero_columns)
                h, w = gray_img.shape[0], gray_img.shape[1]
                # check num pkts
                if h < self.config.num_pkts:
                    # pad with zeros
                    pad_height = self.config.num_pkts - h
                    gray_img = cv2.copyMakeBorder(
                        gray_img, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0
                    )
                elif h > self.config.num_pkts:
                    gray_img = gray_img[: self.config.num_pkts, :]
                # check byte length but not for normalized bcz normalized has only 256 cols
                if not self.config.use_normalized:
                    if w < self.config.byte_length:
                        # pad with zeros
                        pad_width = self.config.byte_length - w
                        gray_img = cv2.copyMakeBorder(
                            gray_img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0
                        )
                    elif w > self.config.byte_length:
                        gray_img = gray_img[:, : self.config.byte_length]
            label = self.label_encoding[label] if self.label_encoding else label

            gray_img = self.scaler(gray_img)
            return gray_img, label, lbl_string

        raise ValueError(f"Image not found at {image_path}.")


class TorchImageDataset(TorchDataset):
    def __init__(self, dataset: DFDataSet):
        self.dataset = dataset
        self.config = dataset.config
        self.data = dataset.data_df
        self.label_encoding = dataset.label_encoding
        self.data_type = dataset.data_type
        self.num_classes = len(self.label_encoding)
        self.label_index = {
            np.array(v).argmax(): k for k, v in self.label_encoding.items()
        }
        # Get class counts
        class_counts = self.data["label"].value_counts().to_dict()

        # Compute inverse frequency weights
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}

        # Order weights according to label encoding
        weights_list = [class_weights[label] for label in self.label_encoding.keys()]

        # Convert to tensor (no need to normalize)
        self.class_weights = torch.tensor(weights_list, dtype=torch.float32)
        self.class_weights = (
            self.class_weights * len(self.class_weights) / self.class_weights.sum()
        )
        self.label_counts = class_counts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, _ = self.dataset[idx]
        self.current_label = self.label_index.get(np.array(label).argmax(), "Unknown")
        self.curr_image_path = self.dataset.curr_image_path
        tensor = torch.from_numpy(image).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)  # C,H,W
        label_tensor = torch.tensor(label, dtype=torch.float)
        return tensor, label_tensor

    def __repr__(self):
        return f"TorchImageDataset(dataset={self.dataset})"

