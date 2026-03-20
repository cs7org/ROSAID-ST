from pathlib import Path
import pandas as pd
from loguru import logger
from rosaid.core.defs import NormalImageType

root = Path("rosaid/results/image_classification")
models = list(root.rglob("*.csv"))
# ("ORIGINAL" "FILTERED" "NORMALIZED" "ZSCORE" "ZGRAM1D" "ZGRAM3D")
normal_types = NormalImageType._member_names_

logger.info(f"Found {len(models)} models to evaluate.")


clf_types = ['binary', 'multiclass']
for clf_type in clf_types:
    results = []
    for model_path in models:
        if clf_type not in str(model_path):
            continue
        model_path = model_path.parent
        logger.info(f"Evaluating model at {model_path}")
        csv_path = model_path / "metrics.csv"
        log_path = model_path / "trainer.log"
        dataset_type = 'DNP3' if 'DNP3' in str(model_path) else 'IEC104' if 'IEC' in str(model_path) else 'ROSIDS23'

        data_type ='ORIGINAL'
        for nt in normal_types:
            if nt in str(model_path):
                data_type = nt
                break
        if data_type is None:
            continue

        is_complete = False
        if log_path.exists():
            with open(log_path, "r") as f:
                for line in f:
                    if "Training complete" in line:
                        is_complete = True
                        break

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if df.empty or len(df)<1:
                logger.warning(f"Metrics CSV is empty for model at {model_path}")
                continue
            best_epoch = df.loc[df['val_f1_score'].idxmax()]
            results.append({
                # "model_path": str(model_path),
                'dataset':dataset_type,
                'data_type':data_type,
                'run_date':model_path.name.split('_')[-1],
                "is_complete": is_complete,
                "best_epoch": int(best_epoch['epoch']),
                "val_accuracy": float(best_epoch['val_accuracy']),
                "val_f1_score": float(best_epoch['val_f1_score']),
                "model_name": model_path.name
            })

    if len(results)==0:
        continue
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['dataset','val_f1_score'], ascending=False)
    output_path = root / f"{clf_type}_image_model_evaluation.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Evaluation results for {clf_type} classification:")
    logger.info(f'\n{results_df.to_string(index=False)}')
    logger.info(f"Saved evaluation results to {output_path}")
    