import torch
from torch import nn


class BaseFeatureModel(nn.Module):
    def get_block_names(self) -> list[str]:
        """Return list of block names for student-teacher training"""
        raise NotImplementedError("Subclasses must implement get_block_names method")

    def get_block(self, block_name: str) -> nn.Module:
        """Get specific block by name for feature matching"""
        raise NotImplementedError("Subclasses must implement get_block method")

    def extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get features from all layer as dict"""
        raise NotImplementedError("Subclasses must implement extract_features method")

    def reset_parameters(self):
        """Reset model parameters for re-initialization"""
        raise NotImplementedError("Subclasses must implement reset_parameters method")
