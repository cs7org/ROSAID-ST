import argparse
from pathlib import Path

import torch
from loguru import logger

from rosaid.core.defs import DNP3_CLASSES, IEC104_CLASSES,ROSIDS23_CLASSES
from rosaid.data.dataset import DFDataSet, SessionImageDataConfig,TorchImageDataset
from rosaid.evaluation.image_evaluation import ImageEvaluator, ImageEvaluatorConfig
import pandas as pd
import matplotlib as mpl
import json 

# Set global font family and size
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
mpl.rcParams["font.size"] = 12  # Adjust per Springer guidelines
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 13
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11

# Optional: PDF/LaTeX text rendering for enhanced output
mpl.rcParams["text.usetex"] = False

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
    "--expt_name",
    type=str,
    default=r"image_classification",
    help="Models are read from [project_dir/results/expt_name].",
)
parser.add_argument(
    "--valid_models",
    type=str,
    default="ROSIDS23_ORIGINAL,ROSIDS23_FILTERED,ROSIDS23_NORMALIZED,ROSIDS23normalized_frequency,ROSIDS23_blockcnn2d__nosampling_binary_20260102",
    help="Type of model to evaluate.",
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
    logger.info(f"Args: {args}")
    
    available_models = list((project_dir / "results" / args.expt_name).rglob("*best_model_full.pth"))

    # valid_models = ['ROSIDS23_ORIGINAL', 'ROSIDS23_FILTERED', 'ROSIDS23_NORMALIZED',
    #                 'ROSIDS23normalized_frequency']
    valid_models = [f.strip() for f in args.valid_models.split(",")]

    logger.info(f"Available models: {available_models}")
    final_results = pd.DataFrame()

    for model_pth in available_models:
        try:
            is_valid = False
            for vm in valid_models:
                if vm in str(model_pth):
                    is_valid = True
                    break
            if not is_valid:
                logger.info(f"Skipping model {model_pth} as it is not in valid models list.")
                continue
            logger.info(f"Evaluating model: {model_pth}")
            # model_pth = Path('rosaid/results/image_classification/ROSIDS23_FILTERED/mobilenet_v3_large_nosampling_binary_20260122/best_model_full.pth')
            model_name = str(model_pth).lower()
            if "dnp3" in model_name:
                data_type = "DNP3"
            elif "iec104" in model_name:
                data_type = "IEC104"
            elif "rosids23" in model_name:
                data_type = "ROSIDS23"
            else:
                raise ValueError(f"Cannot infer data type from model name: {model_name}")
            use_normalized = "normalized" in model_pth.parent.name.lower() 

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
                raise ValueError(f"Unsupported data type: {data_type}")
            expt_name = f"evaluation/{model_pth.parent.parent.name}"
            normal_label = labels[0]
            num_pkts=100

            data_config_path = model_pth.parent / "dataset_config.json"
            if data_config_path.exists():
                with open(data_config_path, 'r') as f:
                    data_config = json.load(f)
                config = SessionImageDataConfig(**data_config)
                config.session_images_dir = data_dir / data_type.upper() / "session_images"
                config.labels_file = data_dir / data_type.upper() / "labelled_sessions.csv"
                logger.info(f"Loaded dataset config from {data_config_path}")
            else:
                logger.warning(f"Dataset config file not found at {data_config_path}. Using default configuration.")
                # Configuration parameters
                config = SessionImageDataConfig(
                    max_data=args.max_data,
                    session_images_dir=data_dir / data_type.upper() / "session_images",
                    labels_file=data_dir / data_type.upper() / "labelled_sessions.csv",
                    use_normalized=False,
                    attack_only=attack_only,
                    labels=labels,
                    normal_label=normal_label,
                    num_pkts=num_pkts,
                    combine_attacks=True,#'binary' in str(model_pth),
                    normal_image_type=1
                )

            # Load the dataset
            train_ds, test_ds = DFDataSet(config=config).load_data()
            img, lbl, lbl_str = test_ds[0]

            expt_dir = project_dir / "results" / expt_name
            if not expt_dir.exists():
                expt_dir.mkdir(parents=True)
            model_name = model_pth.parent.name
            model = torch.load(str(model_pth), weights_only=False, map_location="cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loaded model of type: {model.__class__.__name__} from {model_pth}")
            num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of trainable parameters in the model: {num_parameters}")

            # cv2.imwrite(str(expt_dir / f"{model_name}.png"), (img * 255).astype(np.uint8))

            evaluator_config = ImageEvaluatorConfig(
                batch_size=batch_size,
                output_dir=expt_dir,
                multiclass=len(labels) > 2,
                model_name=model_name,
                is_teacher_classifier=False#'resnet' not in model_name.lower(),
            )
            evaluator = ImageEvaluator(model=model, dataset=TorchImageDataset(test_ds), config=evaluator_config)
            results = evaluator.evaluate()
            results["data_type"] = data_type
            results["model_name"] = model_name
            results["model_type"] = model.__class__.__name__
            results["is_normalized"] = use_normalized

            if final_results.empty:
                final_results = pd.DataFrame(columns=results.keys())
            try:
                # append the results row to final_results
                final_results = pd.concat([final_results, pd.DataFrame([results])], ignore_index=True)
                final_results.to_csv(expt_dir / "final_evaluation_results.csv", index=False)
            except Exception as e:
                logger.error(f"Error while appending results for model {model_name}: {e}")
            logger.info(f"Completed evaluation for model: {model_name}")
        except Exception as e:
            logger.error(f"Error while evaluating model {model_pth}: {e}")

    # now plot final results
    logger.info("Final Evaluation Results:")
    logger.info(f"\n{final_results}")
    logger.remove()
