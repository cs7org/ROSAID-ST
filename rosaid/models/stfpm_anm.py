# SRC: https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/stfpm/torch_model.py
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for Student-Teacher Feature Pyramid Matching.

This module implements the core PyTorch model architecture for the STFPM anomaly
detection method as described in `Wang et al. (2021)
<https://arxiv.org/abs/2103.04257>`_.

The model consists of:
- A pre-trained teacher network that extracts multi-scale features
- A student network that learns to match the teacher's feature representations
- Feature pyramid matching between student and teacher features
- Anomaly detection based on feature discrepancy

Example:
    >>> from anomalib.models.image.stfpm.torch_model import STFPMModel
    >>> model = STFPMModel(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )
    >>> features = model(torch.randn(1, 3, 256, 256))

See Also:
    - :class:`STFPMModel`: Main PyTorch model implementation
    - :class:`STFPMLoss`: Loss function for training
    - :class:`AnomalyMapGenerator`: Anomaly map generation from features
"""

from collections.abc import Sequence

import torch
from torch import nn

from .anomaly_map import AnomalyMapGenerator
from .feature_extractor import TimmFeatureExtractor


class STFPMModel(nn.Module):
    """PyTorch implementation of the STFPM model.

    The Student-Teacher Feature Pyramid Matching model consists of a pre-trained
    teacher network and a student network that learns to match the teacher's
    feature representations. The model detects anomalies by comparing feature
    discrepancies between the teacher and student networks.

    Args:
        layers (Sequence[str]): Names of layers from which to extract features.
            For example ``["layer1", "layer2", "layer3"]``.
        backbone (str, optional): Name of the backbone CNN architecture used for
            both teacher and student networks. Supported backbones can be found
            in timm library. Defaults to ``"resnet18"``.

    Example:
        >>> import torch
        >>> from anomalib.models.image.stfpm.torch_model import STFPMModel
        >>> model = STFPMModel(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2", "layer3"]
        ... )
        >>> input_tensor = torch.randn(1, 3, 256, 256)
        >>> features = model(input_tensor)

    Note:
        The teacher model is initialized with pre-trained weights and frozen
        during training, while the student model is trained from scratch.

    Attributes:
        tiler (Tiler | None): Optional tiler for processing large images in
            patches.
        teacher_model (TimmFeatureExtractor): Pre-trained teacher network for
            feature extraction.
        student_model (TimmFeatureExtractor): Student network that learns to
            match teacher features.
        anomaly_map_generator (AnomalyMapGenerator): Module to generate anomaly
            maps from features.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "resnet18",
        teacher_model: str = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if teacher_model is not None:
            self.teacher_model = torch.load(teacher_model, weights_only=False).eval()
        else:
            self.teacher_model = TimmFeatureExtractor(
                backbone=self.backbone, pre_trained=True, layers=layers
            ).eval()
        self.student_model = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=False,
            layers=layers,
            requires_grad=True,
        )

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator()

    def fix_channel(self, images: torch.Tensor) -> torch.Tensor:
        """Fix input channel size to 3 by duplicating channels if needed.

        Args:
            images (torch.Tensor): Input image tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Image tensor with 3 channels.
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return images

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass through teacher and student networks.

        The forward pass behavior differs between training and evaluation:
        - Training: Returns features from both teacher and student networks
        - Evaluation: Returns anomaly maps generated from feature differences

        Args:
            images (torch.Tensor): Batch of input images with shape
                ``(N, C, H, W)``.

        Returns:
            Training mode:
                tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                    Features from teacher and student networks respectively.
                    Each dict maps layer names to feature tensors.
            Evaluation mode:
                InferenceBatch:
                    Batch containing anomaly maps and prediction scores.

        Example:
            >>> import torch
            >>> from anomalib.models.image.stfpm.torch_model import STFPMModel
            >>> model = STFPMModel(layers=["layer1", "layer2", "layer3"])
            >>> input_tensor = torch.randn(1, 3, 256, 256)
            >>> # Training mode
            >>> model.train()
            >>> teacher_feats, student_feats = model(input_tensor)
            >>> # Evaluation mode
            >>> model.eval()
            >>> predictions = model(input_tensor)
        """
        images = self.fix_channel(images)

        teacher_features: dict[str, torch.Tensor] = self.teacher_model(images)
        student_features: dict[str, torch.Tensor] = self.student_model(images)

        return teacher_features, student_features

    def inference(
        self, images: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ]:
        output_size = images.shape[-2:]
        images = self.fix_channel(images)
        teacher_features, student_features = (
            self.teacher_model(images),
            self.student_model(images),
        )
        anomaly_map = self.anomaly_map_generator(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score, anomaly_map, (teacher_features, student_features)
