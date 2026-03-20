import cv2
import numpy as np
import torch
from loguru import logger
from pathlib import Path
from rosaid.data.dataset import TorchImageDataset
from pydantic import BaseModel, Field
from rosaid.utils.confusion_matrix import get_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
from tqdm import tqdm
from rosaid.models.stfpm import STFPModel
from rosaid.models.cnn import BlockCNN2D
from rosaid.models.image_model import ImageClfModel
from collections import defaultdict
import torch.nn as nn
import pandas as pd
import sys

class ImageEvaluatorConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1_score"],
        description="List of metrics to compute during evaluation.",
    )
    output_dir: Path = Field(
        default=Path("./evaluation_results"),
        description="Directory to save evaluation results.",
    )
    multiclass: bool = Field(
        default=True,
        description="Whether the classification task is multiclass or binary.",
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for evaluation.",
    )
    model_name: str = Field(
        default="model",
        description="Name of the model being evaluated.",
    )
    is_teacher_classifier: bool = Field(
        default=False,
        description="Whether the model is a teacher classifier.",
    )

class ImageEvaluator:
    def __init__(self, model:STFPModel|torch.nn.Module, dataset: TorchImageDataset, config: ImageEvaluatorConfig):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.output_dir = config.output_dir
        self.multiclass = config.multiclass
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.multiclass = config.multiclass
        self.model_name = config.model_name
        self.output_dir = self.output_dir / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = defaultdict(list)
        self.written_labels = set()
        
        
    def _get_outputs(self, images: torch.Tensor,label_name:str|None=None) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, STFPModel):
                pred_score, anomaly_map, (teacher_features, student_features) = self.model.inference(images)
                # print('teacher_features keys:', teacher_features.keys())
                #save sample image and anomaly map in png format
                if label_name is not None and label_name not in self.written_labels:
                    self.written_labels.add(label_name)
                    anomaly_map_np = anomaly_map.cpu().numpy().squeeze()
                    norm = cv2.normalize(
                        anomaly_map_np, None, 0, 255, cv2.NORM_MINMAX
                    )
                    vis_map = cv2.applyColorMap(
                        norm.astype(np.uint8), cv2.COLORMAP_JET
                    )
                    cv2.imwrite(str(self.output_dir / f"{label_name}_input.png"), (images[0,0].cpu().numpy() * 255).astype(np.uint8))
                    cv2.imwrite(str(self.output_dir / f"{label_name}_anomaly_map.png"), vis_map)
                
                # mse between features
                for feat_name in teacher_features.keys():
                    tfeat = teacher_features[feat_name]
                    sfeat = student_features[feat_name]
                    feat_mse = nn.MSELoss()(sfeat, tfeat).item()
                    self.metrics[f"feature_mse_{feat_name}"].append(feat_mse)

                self.metrics["anomaly_score"].extend(pred_score.cpu().numpy().tolist()[0])
                self.metrics['avg_mse'].append(np.mean([self.metrics[f"feature_mse_{feat_name}"][-1] for feat_name in teacher_features.keys()]))
                if self.config.is_teacher_classifier:
                    # teacher scores
                    tlogits, tprobs = self.model.teacher(images)
                    slogits, sprobs = self.model.student(images)
                    self.metrics["teacher_score"].extend(torch.argmax(tprobs, dim=1).cpu().numpy().tolist())
                    self.metrics["student_score"].extend(torch.argmax(sprobs, dim=1).cpu().numpy().tolist())
                

            elif isinstance(self.model, BlockCNN2D) or isinstance(self.model, ImageClfModel):
                logits, probs = self.model(images)
                if logits.shape[1] == 1:
                    # binary classification
                    preds = (probs > 0.5).long()
                    self.metrics["pred_score"].extend(preds.cpu().numpy().tolist()[0])
                else:
                    self.metrics["pred_score"].extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

            else:
                pred_score, anomaly_map, (teacher_features, student_features) = self.model.inference(images)
                #save sample image and anomaly map in png format
                if label_name is not None and label_name not in self.written_labels:
                    self.written_labels.add(label_name)
                    anomaly_map_np = anomaly_map.cpu().numpy().squeeze()
                    norm = cv2.normalize(
                        anomaly_map_np, None, 0, 255, cv2.NORM_MINMAX
                    )
                    vis_map = cv2.applyColorMap(
                        norm.astype(np.uint8), cv2.COLORMAP_JET
                    )
                    cv2.imwrite(str(self.output_dir / f"{label_name}_input.png"), (images[0,0].cpu().numpy() * 255).astype(np.uint8))
                    cv2.imwrite(str(self.output_dir / f"{label_name}_anomaly_map.png"), vis_map)
                
                # mse between features
                for feat_name in teacher_features.keys():
                    tfeat = teacher_features[feat_name]
                    sfeat = student_features[feat_name]
                    feat_mse = nn.MSELoss()(sfeat, tfeat).item()
                    self.metrics[f"feature_mse_{feat_name}"].append(feat_mse)
                
                self.metrics['avg_mse'].append(np.mean([self.metrics[f"feature_mse_{feat_name}"][-1] for feat_name in teacher_features.keys()]))
                self.metrics["anomaly_score"].extend(pred_score.cpu().numpy().tolist()[0])

        return self.metrics

    def evaluate(self):
        self.model.eval()
        
        pbar = tqdm(self.dataset, desc=f"Eval(model={self.model_name})", unit="item",
                    disable=not sys.stdout.isatty(),  )
        with torch.no_grad():
            for images, labels in pbar:
                data_name = self.dataset.curr_image_path.name
                label_name = self.dataset.current_label
                images = images.to(self.device).unsqueeze(0).float()
                labels = labels.to(self.device).unsqueeze(0).float()
                # logger.debug(f"Evaluating batch with images shape: {images.shape}, labels shape: {labels.shape}")
                # returns logits, probs
                # _,outputs = self.model(images)
                self.metrics = self._get_outputs(images, label_name)
                self.metrics['data_name'].append(data_name)
                self.metrics['true_value'].append(labels.cpu().numpy().argmax(axis=1)[0])
                if self.config.is_teacher_classifier:
                    true_value = self.metrics['true_value'][-1]
                    pred_value = self.metrics.get('pred_score', self.metrics.get('teacher_score', []))[-1]

                    # log true and pred labels
                    pbar.set_postfix({"true": true_value, "pred": pred_value})
        final_metrics = {}
        true_labels = self.metrics['true_value']
        if 'pred_score' in self.metrics:
            pred_labels = self.metrics['pred_score']
        elif 'teacher_score' in self.metrics:
            pred_labels = self.metrics['teacher_score']
        elif 'student_score' in self.metrics:
            pred_labels = self.metrics['student_score']
        
        # save self.metrics to a csv file
        # logger.info(f"{self.metrics}")
        results_df = pd.DataFrame(self.metrics)
        if self.dataset.config.filter_first_nonzero_columns:
            results_df['filtered_columns'] = True
        results_df.to_csv(self.output_dir / f"detailed_results_{self.model_name}.csv", index=False)
        logger.info(f"Saved detailed results to {self.output_dir / f'detailed_results_{self.model_name}.csv'}")
        if 'true_value' in results_df.columns and 'pred_score' in results_df.columns:
            auc = 0.0
            try:
                
                true_value = results_df['true_value'].values
                pred_value = results_df['pred_score'].values
                accuracy = accuracy_score(true_value, pred_value)
                macro_f1 = f1_score(true_value, pred_value, average='macro')
                micro_f1 = f1_score(true_value, pred_value, average='micro')
               
                with open(self.output_dir / f"metrics_{self.model_name}.txt", "w") as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"Micro F1 Score: {micro_f1}\n")
                    f.write(f"Macro F1 Score: {macro_f1}\n")
               
                logger.info(f"Accuracy: {accuracy}, Micro F1 Score: {micro_f1}, Macro F1 Score: {macro_f1}")
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC Score: {e}")

        
        if self.config.is_teacher_classifier:
            f1,cm = get_confusion_matrix(
            predictions=pred_labels,
            targets=true_labels,
            label_keys=self.dataset.label_encoding,
            out_file=self.output_dir / f"cm_{self.model_name}.png",
        )
            final_metrics["accuracy"] = accuracy_score(true_labels, pred_labels)
            for average in ['weighted','micro','macro']:
                key = f"{average}_f1_score"
                final_metrics[key] = f1_score(true_labels, pred_labels, average=average)
                key = f"{average}_precision"
                final_metrics[key] = precision_score(true_labels, pred_labels, average=average)
                key = f"{average}_recall"
                final_metrics[key] = recall_score(true_labels, pred_labels, average=average)
            logger.info(f"Evaluation metrics: {final_metrics}")
            final_metrics["confusion_matrix"] = cm.tolist()
            logger.info(f"Confusion matrix\n{cm}\n saved to {self.output_dir / f'cm_{self.model_name}.png'} with F1 score: {f1}")
        # save final metrics to a json file
        with open(self.output_dir / f"metrics_{self.model_name}.json", "w") as f:
            json.dump(final_metrics, f, indent=4)
        return final_metrics