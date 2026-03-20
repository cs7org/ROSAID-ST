# CLF/STFPM Prediction on:
    # clean image: done with image_model_evaluation.py
    # full adversarial: do here
    # masked adversarial
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from rosaid.core.defs import ROSIDS23_CLASSES
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict

# argument parser to accept: image_type[normalized,normal], backbone, max_data
parser = argparse.ArgumentParser(description="Session Image Trainer Configuration")

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
    "--clf_model_path",
    type=str,
    default=r"rosaid\models\clf_model.pth",
    help="Path to the classifier model.",
)
parser.add_argument(
    "--stfpm_model_path",
    type=str,
    default=r"rosaid\models\stfpm_model.pth",
    help="Path to the STFPM model.",
)


parser.add_argument(
    "--attack_only",
    action="store_true",
    default=False,
    help="Use only attack samples for training.",
)

class DataSet(torch.utils.data.Dataset):

    def __init__(
        self,
        input_dir: Path,
        labels: list[str],
        data_split: str,
        adversarial_type: str,
        max_samples: int = -1000,
    ):

        self.input_dir = input_dir
        self.labels = labels
        self.data_split = data_split
        self.adversarial_type = adversarial_type
        self.npz_files = list(input_dir.rglob(f"{adversarial_type}/{data_split}/*.npz"))
        self.max_samples = max_samples

        if not self.npz_files:
            raise FileNotFoundError(
                f"No .npz files found in {input_dir / adversarial_type / data_split}"
            )
        refined_npz_files = []
        for label in self.labels:
            label_files = [
                f for f in self.npz_files if label.lower() in f.name.lower()
            ]
            if not label_files:
                logger.warning(f"No files found for label {label} in {adversarial_type}")
            else:
                refined_npz_files.extend(label_files)
        self.npz_files = refined_npz_files

    def __len__(self):
        return (
            len(self.npz_files)
            if self.max_samples < 0
            else min(len(self.npz_files), self.max_samples)
        )

    def __getitem__(self, idx):
        selected_file = self.npz_files[idx]
        npz_data = np.load(selected_file)
        image_name = selected_file.name
        input_image = npz_data["inputs"]
        adversarial_image = npz_data["adversarial"]
        masked_adv = npz_data["masked"]
        label = npz_data["label_str"].item()

        cv2.imwrite(
            "rosaid/res.png",
            adversarial_image,
        )

        label_index = self.labels.index(label)
        img_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0) / 255
        adv_tensor = (
            torch.tensor(adversarial_image, dtype=torch.float32).unsqueeze(0) / 255
        )
        masked_adv_tensor = (
            torch.tensor(masked_adv, dtype=torch.float32).unsqueeze(0) / 255
        )
        self.curr_label = label

        return img_tensor, adv_tensor, masked_adv_tensor, label_index, image_name


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    data_dir = Path(args.data_dir)
    project_dir = Path(args.project_dir)
    attack_only = args.attack_only
    clf_model = Path(args.clf_model_path) / 'best_model_full.pth'
    stfpm_model = Path(args.stfpm_model_path) / 'best_model_full.pth'
    logger.info(f"Args: {args}")
    out_dir = project_dir / "results" /'evaluation'/'adversarial_evaluation'/f"{stfpm_model.parent.name}_{clf_model.parent.name}_results"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    
    # load full model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_model = torch.load(str(clf_model), map_location=device, weights_only=False)
    stfpm_model = torch.load(str(stfpm_model), map_location=device, weights_only=False)

    
    adversarial_types = [pth.name for pth in data_dir.glob("*") if pth.is_dir()]
    classes=ROSIDS23_CLASSES
    binary_labels = [classes[0], "ATTACK"]
                
    loss = nn.MSELoss()

    # adversarial:[label1, label2]
    saved_sample = defaultdict(list)
    saved_adv_samples = defaultdict(list)

    for adversarial_type in adversarial_types:
        classifier_results = pd.DataFrame()
        stfpm_results = pd.DataFrame()
        # only do momentum attacks for now
        # if 'mom' not in adversarial_type.lower():
        #     continue
        results_dir = out_dir / adversarial_type
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        logger.info(f"Evaluating adversarial type: {adversarial_type}")
        dataset = DataSet(
            input_dir=data_dir,
            labels=classes,
            data_split="val",
            adversarial_type=adversarial_type,
            max_samples=args.max_data,
        )
        logger.info(f"Number of samples in dataset: {len(dataset)}")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            imgs, adv_imgs, masked_adv_imgs, labels, img_names = batch
            imgs = imgs.to(device)
            adv_imgs = adv_imgs.to(device)
            # recreate masked_adv_imgs: zero out columns that are zeroed in all img in imgs
            new_masked_adv_imgs = torch.zeros_like(adv_imgs)
            for i in range(imgs.shape[0]):
                zero_cols = (imgs[i].sum(dim=1) == 0)
                new_masked_adv_imgs[i] = adv_imgs[i] * (~zero_cols).unsqueeze(1)
            masked_adv_imgs = new_masked_adv_imgs.to(device)
            labels = labels.to(device)

            
            # Classifier predictions
            with torch.no_grad():
                _,clf_outputs = clf_model(imgs)
                _,adv_clf_outputs = clf_model(adv_imgs)
                _,masked_adv_clf_outputs = clf_model(masked_adv_imgs)

                if clf_outputs.shape[1] == 1:
                    # binary classification
                    clf_outputs = torch.zeros((clf_outputs.shape[0], 2), device=clf_outputs.device).scatter_(1, (clf_outputs>0.5).long(), 1)
                    adv_clf_outputs = torch.zeros((adv_clf_outputs.shape[0], 2), device=adv_clf_outputs.device).scatter_(1, (adv_clf_outputs>0.5).long(), 1)
                    masked_adv_clf_outputs = torch.zeros((masked_adv_clf_outputs.shape[0], 2), device=masked_adv_clf_outputs.device).scatter_(1, (masked_adv_clf_outputs>0.5).long(), 1)

            _, clf_preds = torch.max(clf_outputs, 1)
            _, adv_clf_preds = torch.max(adv_clf_outputs, 1)
            _, masked_adv_clf_preds = torch.max(masked_adv_clf_outputs, 1)

            # STFPM predictions
            with torch.no_grad():
                pred_score, anomaly_map, (teacher_features, student_features) = stfpm_model.inference(imgs)
                adv_pred_score, adv_anomaly_map, (adv_teacher_features, adv_student_features) = stfpm_model.inference(adv_imgs)
                masked_adv_pred_score, masked_adv_anomaly_map, (masked_adv_teacher_features, masked_adv_student_features) = stfpm_model.inference(masked_adv_imgs)

            # Collect results
            for i in range(len(labels)):
                true_binary_label = classes[labels[i].item()]==classes[0]
                true_binary_label = classes[0] if true_binary_label else "ATTACK"
                
                lbl = classes[labels[i].item()]
                classifier_results = pd.concat([
                    classifier_results,
                    pd.DataFrame([{
                        "image_name": img_names[i],
                        "true_multi_class_label": classes[labels[i].item()],
                        "true_binary_label": true_binary_label,
                        "adversarial_type": adversarial_type,
                        "clf_pred_clean": binary_labels[clf_preds[i].item()],
                        "clf_pred_adv": binary_labels[adv_clf_preds[i].item()],
                        "clf_pred_masked_adv": binary_labels[masked_adv_clf_preds[i].item()],
                    }])
                ], ignore_index=True)

                stfpm_res = dict(
                    image_name=img_names[i],
                    true_multi_class_label=classes[labels[i].item()],
                    true_binary_label=true_binary_label,
                    adversarial_type=adversarial_type,
                    clean_anomaly_score=pred_score[i].item(),
                    adv_anomaly_score=adv_pred_score[i].item(),
                    masked_adv_anomaly_score=masked_adv_pred_score[i].item(),
                )
                for feat_name in teacher_features.keys():
                    stfpm_res[f"clean_feature_mse_{feat_name}"] = loss(student_features[feat_name][i], teacher_features[feat_name][i]).item()
                    stfpm_res[f"adv_feature_mse_{feat_name}"] = loss(adv_student_features[feat_name][i], adv_teacher_features[feat_name][i]).item()
                    stfpm_res[f"masked_adv_feature_mse_{feat_name}"] = loss(masked_adv_student_features[feat_name][i], masked_adv_teacher_features[feat_name][i]).item()
                stfpm_results = pd.concat([
                    stfpm_results,
                    pd.DataFrame([stfpm_res])
                ], ignore_index=True)

                # Save anomaly maps for first occurrence of each class
                if lbl not in saved_adv_samples[adversarial_type]:
                    saved_sample[adversarial_type].append(lbl)
                    sample_idx = (labels == classes.index(lbl)).nonzero(as_tuple=True)[0][0]
                    sample_img = imgs[sample_idx].cpu().squeeze().numpy() * 255
                    sample_adv_img = adv_imgs[sample_idx].cpu().squeeze().numpy() * 255
                    sample_masked_adv_img = masked_adv_imgs[sample_idx].cpu().squeeze().numpy() * 255

                    cv2.imwrite(
                        str(results_dir / f"{lbl}_clean.png"),
                        sample_img.astype(np.uint8),
                    )
                    cv2.imwrite(
                        str(results_dir / f"{lbl}_adv.png"),
                        sample_adv_img.astype(np.uint8),
                    )
                    cv2.imwrite(
                        str(results_dir / f"{lbl}_masked_adv.png"),
                        sample_masked_adv_img.astype(np.uint8),
                    )
                    logger.info(f"Saved sample images for label {lbl} under adversarial type {adversarial_type}")


                    saved_adv_samples[adversarial_type].append(lbl)
                    sample_idx = (labels == classes.index(lbl)).nonzero(as_tuple=True)[0][0]

                    clean_amap = anomaly_map[sample_idx].cpu().squeeze().numpy()
                    adv_amap = adv_anomaly_map[sample_idx].cpu().squeeze().numpy()
                    masked_amap = masked_adv_anomaly_map[sample_idx].cpu().squeeze().numpy()

                    # save non -normalized maps as npz
                    np.savez_compressed(
                        str(results_dir / f"{lbl}_anomaly_maps.npz"),
                        clean_amap=clean_amap,
                        adv_amap=adv_amap,
                        masked_amap=masked_amap,
                        anomaly_scores=np.array([
                            pred_score[sample_idx].item(),
                            adv_pred_score[sample_idx].item(),
                            masked_adv_pred_score[sample_idx].item(),
                        ]),
                    )

                    # Normalize for visualization
                    clean_map = cv2.normalize(clean_amap, None, 0, 255, cv2.NORM_MINMAX)
                    norm_map = cv2.normalize(adv_amap, None, 0, 255, cv2.NORM_MINMAX)
                    norm_masked_map = cv2.normalize(masked_amap, None, 0, 255, cv2.NORM_MINMAX)

                    # now use jet colormap to save anomaly maps
                    clean_map = cv2.applyColorMap(clean_map.astype(np.uint8), cv2.COLORMAP_JET)
                    norm_map = cv2.applyColorMap(norm_map.astype(np.uint8), cv2.COLORMAP_JET)
                    norm_masked_map = cv2.applyColorMap(norm_masked_map.astype(np.uint8), cv2.COLORMAP_JET)

                    cv2.imwrite(
                        str(results_dir / f"{lbl}_clean_amap.png"),
                        clean_map.astype(np.uint8),
                    )
                    cv2.imwrite(
                        str(results_dir / f"{lbl}_adv_amap.png"),
                        norm_map.astype(np.uint8),
                    )
                    cv2.imwrite(
                        str(results_dir / f"{lbl}_masked_adv_amap.png"),
                        norm_masked_map.astype(np.uint8),
                    )

                    logger.info(f"Saved adversarial anomaly maps for label {lbl} under adversarial type {adversarial_type}")



        # save results to csv
        clf_res_pth = results_dir / "adversarial_classifier_results.csv"
        stfpm_res_pth = results_dir / "adversarial_stfpm_results.csv"
        classifier_results.to_csv(clf_res_pth, index=False)
        stfpm_results.to_csv(stfpm_res_pth, index=False)
        logger.info(f"Saved classifier results to {clf_res_pth}")
        logger.info(f"Saved STFPM results to {stfpm_res_pth}")
