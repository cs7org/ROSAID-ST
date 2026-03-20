"""
- We have already saved pkts in images.
- Now we will make all the packets with length 3*23*23.
- Each session will be represented as a set of images (num_pkts, 3, 23, 23)

"""

from pathlib import Path

import cv2
import numpy as np
import torch

from rosaid.data.dataset import DataType, DFDataSet, TorchImageDataset
from rosaid.utils.image import (
    get_filtered_image,
)


class SessionDFDataset(DFDataSet):
    height: int = 23
    width: int = 23
    channels: int = 3
    byte_length: int = channels * height * width

    def get_datasets(self):
        train_dataset = SessionDFDataset(self.config.copy())
        train_dataset.data_df = self.train_df
        train_dataset.data_type = DataType.TRAIN
        train_dataset.label_encoding = self.label_encoding

        test_dataset = SessionDFDataset(self.config.copy())
        test_dataset.data_df = self.test_df
        test_dataset.data_type = DataType.VALIDATION
        test_dataset.label_encoding = self.label_encoding

        return train_dataset, test_dataset

    def __getitem__(self, idx):
        if self.data_df is not None:
            row = self.data_df.iloc[idx]
        image_path = Path(row["file_path"])
        label = row["label"]
        lbl_string = row.label

        if image_path.exists():
            img = cv2.imread(str(image_path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = get_filtered_image(gray_img)
            h, w = gray_img.shape

            if not self.config.use_normalized:
                if w < self.byte_length:
                    # pad with zeros
                    pad_width = self.byte_length - w
                    gray_img = cv2.copyMakeBorder(
                        gray_img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0
                    )
                elif w > self.byte_length:
                    gray_img = gray_img[:, : self.byte_length]
            if h == 1:
                # if only one row, pad with zeros to make 2 rows
                gray_img = cv2.copyMakeBorder(
                    gray_img, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0
                )
            label = self.label_encoding[label] if self.label_encoding else label
            packets = []

            gray_img = self.scaler(gray_img)
            for packet in gray_img:
                packet = packet.reshape(self.height, self.width, self.channels)
                packets.append(packet)

            gray_img = np.array(packets)
            if len(gray_img) > self.config.num_pkts and self.config.num_pkts > 0:
                gray_img = gray_img[: self.config.num_pkts]
            return gray_img, label, lbl_string

        raise ValueError(f"Image not found at {image_path}.")


class TorchSessionImageDataset(TorchImageDataset):
    def __getitem__(self, idx):
        image, label, _ = self.dataset[idx]
        self.current_label = self.label_index.get(np.array(label).argmax(), "Unknown")
        tensor = torch.from_numpy(image).float().permute(0, 3, 1, 2)  # B,C,H,W
        label_tensor = torch.tensor(label, dtype=torch.float)
        return tensor, label_tensor


class DynamicSessionDFDataset(DFDataSet):
    def get_datasets(self):
        train_dataset = DynamicSessionDFDataset(self.config.copy())
        train_dataset.data_df = self.train_df
        train_dataset.data_type = DataType.TRAIN
        train_dataset.label_encoding = self.label_encoding

        test_dataset = DynamicSessionDFDataset(self.config.copy())
        test_dataset.data_df = self.test_df
        test_dataset.data_type = DataType.VALIDATION
        test_dataset.label_encoding = self.label_encoding

        return train_dataset, test_dataset

    def __getitem__(self, idx):
        if self.data_df is not None:
            row = self.data_df.iloc[idx]
        image_path = Path(row["file_path"])
        label = row["label"]
        lbl_string = row.label

        if image_path.exists():
            img = cv2.imread(str(image_path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = get_filtered_image(gray_img)
            h, w = gray_img.shape

            label = self.label_encoding[label] if self.label_encoding else label
            packets = []


            gray_img = self.scaler(gray_img)

            if self.config.num_pkts > 0:
                gray_img = gray_img[: self.config.num_pkts]

            for packet in gray_img:
                # find the right most index which is not zero
                right_most = (
                    np.where(packet != 0)[0].max() if np.any(packet != 0) else 0
                )
                packet = packet[: right_most + 1].reshape(1, -1)
                packets.append(packet)

            # gray_img = np.array(packets)
            return gray_img, packets, label, lbl_string

        raise ValueError(f"Image not found at {image_path}.")


class DynamicTorchSessionImageDataset(TorchImageDataset):
    def __getitem__(self, idx):
        gray_img, packets, label, _ = self.dataset[idx]
        self.current_label = self.label_index.get(np.array(label).argmax(), "Unknown")
        packets_tensors = [torch.from_numpy(pkt).float() for pkt in packets]
        label_tensor = torch.tensor(label, dtype=torch.float)
        return gray_img, packets_tensors, label_tensor
