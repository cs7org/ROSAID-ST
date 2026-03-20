# CLF/STFPM Prediction on:
    # clean image: done with image_model_evaluation.py
    # full adversarial: do here
    # masked adversarial
import argparse
from pathlib import Path

import torch
from loguru import logger

from rosaid.core.defs import DNP3_CLASSES, IEC104_CLASSES,ROSIDS23_CLASSES
from rosaid.data.dataset import DFDataSet, SessionImageDataConfig,TorchImageDataset
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
    # default=r"rosaid\data\120_timeout_dnp3_sessions",
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



if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    data_dir = Path(args.data_dir)
    project_dir = Path(args.project_dir)
    attack_only = args.attack_only
    clf_model = Path(args.clf_model_path) / 'best_model_full.pth'
    stfpm_model = Path(args.stfpm_model_path) / 'best_model_full.pth'
    logger.info(f"Args: {args}")
    out_dir = project_dir / "results" /'evaluation'/'OOD_evaluation'/f"{stfpm_model.parent.name}_{clf_model.parent.name}_results"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    data_types = data_dir.iterdir()
    # load full model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_model = torch.load(str(clf_model), map_location=device, weights_only=False)
    stfpm_model = torch.load(str(stfpm_model), map_location=device, weights_only=False)

    
    for data_type_dir in data_types:
        data_type = data_type_dir.name.upper()
        logger.info(f"Processing data type: {data_type}")
    
        # Determine labels and num_pkts based on data_type

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
            # but those seem to be outliers and 300 could be sufficient
            num_pkts = 300
        else:
            Warning(f"Unsupported data type: {data_type}, defaulting to ROSIDS23 settings.")
            continue
        # use 100 for all
        num_pkts = 100
        normal_label = labels[0]
        config = SessionImageDataConfig(
                max_data=args.max_data,
                session_images_dir=data_dir / data_type.upper() / "session_images",
                labels_file=data_dir / data_type.upper() / "labelled_sessions.csv",
                use_normalized=False,
                attack_only=attack_only,
                labels=labels,
                normal_label=normal_label,
                num_pkts=num_pkts,
                combine_attacks='binary' in str(clf_model)
            )

        # Load the dataset
        train_ds, test_ds = DFDataSet(config=config).load_data()
        img, lbl, lbl_str = test_ds[0]
        
        
        classes=labels
        binary_labels = [classes[0], "ATTACK"]
                    
        loss = nn.MSELoss()

        # adversarial:[label1, label2]
        saved_sample = defaultdict(list)
        saved_adv_samples = defaultdict(list)

        classifier_results = pd.DataFrame()
        stfpm_results = pd.DataFrame()
        # only do momentum attacks for now
        # if 'mom' not in adversarial_type.lower():
        #     continue
        results_dir = out_dir / data_type
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        logger.info(f"Evaluating OOD on data type {data_type}")
        # Configuration parameters
        
        dataloader = DataLoader(TorchImageDataset(test_ds), batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            
            # Classifier predictions
            with torch.no_grad():
                _,clf_outputs = clf_model(imgs)

                if clf_outputs.shape[1] == 1:
                    # binary classification
                    clf_outputs = torch.zeros((clf_outputs.shape[0], 2), device=clf_outputs.device).scatter_(1, (clf_outputs>0.5).long(), 1)

            _, clf_preds = torch.max(clf_outputs, 1)
           
            # STFPM predictions
            with torch.no_grad():
                pred_score, anomaly_map, (teacher_features, student_features) = stfpm_model.inference(imgs)
                
            # Collect results
            for i in range(len(labels)):
                true_binary_label = classes[labels[i].argmax().item()]==classes[0]
                true_binary_label = classes[0] if true_binary_label else "ATTACK"

                lbl = classes[labels[i].argmax().item()]
                classifier_results = pd.concat([
                    classifier_results,
                    pd.DataFrame([{
                        "image_idx": i,
                        "true_multi_class_label": classes[labels[i].argmax().item()],
                        "true_binary_label": true_binary_label,
                        "clf_pred_clean": binary_labels[clf_preds[i].item()]
                        }])
                ], ignore_index=True)

                stfpm_res = dict(
                    image_idx=i,
                    true_multi_class_label=classes[labels[i].argmax().item()],
                    true_binary_label=true_binary_label,
                    anomaly_score=pred_score[i].item()
                )
                for feat_name in teacher_features.keys():
                    stfpm_res[f"clean_feature_mse_{feat_name}"] = loss(student_features[feat_name][i], teacher_features[feat_name][i]).item()
                    
                stfpm_results = pd.concat([
                    stfpm_results,
                    pd.DataFrame([stfpm_res])
                ], ignore_index=True)

                
        # save results to csv
        clf_res_pth = results_dir / "classifier_results.csv"
        stfpm_res_pth = results_dir / "stfpm_results.csv"
        classifier_results.to_csv(clf_res_pth, index=False)
        stfpm_results.to_csv(stfpm_res_pth, index=False)
        logger.info(f"Saved classifier results to {clf_res_pth}")
        logger.info(f"Saved STFPM results to {stfpm_res_pth}")
