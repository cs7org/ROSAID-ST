import torch
import torch.nn as nn
from loguru import logger

from rosaid.models.base import BaseFeatureModel

from .anomaly_map import AnomalyMapGenerator


class STFPModel(nn.Module):
    def __init__(
        self,
        student: BaseFeatureModel,
        teacher: BaseFeatureModel,
        layers: list[str],
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.layers = layers
        self.teacher.eval()  # Teacher is not trained
        self.anomaly_map_generator = AnomalyMapGenerator()
        logger.info("Initialized STFPModel with student and teacher.")

    def forward(
        self, x
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        with torch.no_grad():
            teacher_features, teacher_logits = self.teacher.extract_features(x)

        student_features, student_logits = self.student.extract_features(x)

        # feature_losses = {}
        # for layer in self.layers:
        #     t_feat = teacher_features[layer]
        #     s_feat = student_features[layer]
        #     feature_losses[layer] = nn.MSELoss()(s_feat, t_feat)
        new_teacher_features = {}
        new_student_features = {}
        for layer in self.layers:
            new_teacher_features[layer] = teacher_features[layer]
            new_student_features[layer] = student_features[layer]

        return new_teacher_features, new_student_features

    def inference(
        self, images: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ]:
        output_size = images.shape[-2:]
        teacher_features, student_features = self.forward(images)
        anomaly_map = None
        if len(images.shape) == 4:
            anomaly_map = self.anomaly_map_generator(
                teacher_features=teacher_features,
                student_features=student_features,
                image_size=output_size,
            )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score, anomaly_map, (teacher_features, student_features)

