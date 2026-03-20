import cv2
import torch
from loguru import logger

from rosaid.data.dataset import TorchImageDataset
from rosaid.loss.stfpm_loss import STFPMLoss
from rosaid.models.stfpm import STFPMModel
from rosaid.trainers.trainer_stfpm import STFPMTrainer as BaseSTFPMTrainer, STFPMTrainerConfig


class STFPMTrainer(BaseSTFPMTrainer):
    def __init__(
        self,
        config: STFPMTrainerConfig,
        model: STFPMModel,
        train_dataset: TorchImageDataset,
        val_dataset: TorchImageDataset,
        criterion=STFPMLoss(),
    ):
        super().__init__(config, model, train_dataset, val_dataset, criterion)

        self.model.to(self.device)

    # def at_batch_end(self):
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def at_epoch_end(self):
        # plot one image of all labels
        import numpy as np
        from rosaid.utils.vis import plt, subplot_images

        self.model.eval()
        with torch.no_grad():
            all_labels = [k for k in self.val_dataset.label_counts.keys()]
            all_images = []
            all_titles = []
            for curr_lbl in all_labels:
                while True:
                    random_idx = torch.randint(0, len(self.val_dataset), (1,)).item()
                    inputs, label_tensor = self.val_dataset[random_idx]

                    label = self.val_dataset.current_label
                    if label == curr_lbl:
                        inputs = inputs.unsqueeze(0).to(self.device)
                        (
                            pred_score,
                            anomaly_map,
                            (teacher_features, student_features),
                        ) = self.model.inference(inputs)

                        error = self.criterion(teacher_features, student_features)
                        # reverse normalize and uint8 numpy
                        inputs_np = (inputs.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )
                        anomaly_map_np = anomaly_map.cpu().numpy().squeeze()
                        norm = cv2.normalize(
                            anomaly_map_np, None, 0, 255, cv2.NORM_MINMAX
                        )
                        vis_map = cv2.applyColorMap(
                            norm.astype(np.uint8), cv2.COLORMAP_JET
                        )

                        all_images.append(inputs_np)
                        all_images.append(vis_map)
                        all_titles.append(f"Input: ({curr_lbl})")
                        all_titles.append(
                            f"Map (Err, Score): ({error.item():.4f}, {pred_score.item():.4f})"
                        )
                        break
            fig = subplot_images(
                all_images,
                titles=all_titles,
                fig_size=(10, 15),
                order=(len(all_labels), 2),
                axis=False,
                show=False,
            )
            out_dir = self.config.run_dir / "progress_images"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"epoch_{self.epoch}.png")
            plt.close(fig)

            logger.info(
                f"Saved anomaly images for epoch {self.epoch} to {out_dir / f'epoch_{self.epoch}.png'}"
            )
