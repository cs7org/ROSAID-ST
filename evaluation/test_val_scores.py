import pandas as pd
from pathlib import Path
from rosaid.core.defs import NormalImageType

train_root=Path('rosaid/results/image_classification')

metrics_files = list(train_root.rglob('*/metrics.csv'))
results = []
for metrics_file in metrics_files:
    df = pd.read_csv(metrics_file)
    val_f1_scores = df['val_f1_score'].values
    best_epoch_idx = val_f1_scores.argmax()
    best_epoch = df.iloc[best_epoch_idx]
    dataset = 'DNP3' if 'DNP3' in str(metrics_file) else 'IEC104' if 'IEC' in str(metrics_file) else 'ROSIDS23'
    model_name = 'BlockCNN2D' if 'blockcnn2d' in str(metrics_file) else 'MobileNetV3Large' if 'mobilenet' in str(metrics_file) else 'ResNet18'
    data_type = None
    for nt in NormalImageType._member_names_:
        if nt in str(metrics_file):
            data_type = nt
            break
    if data_type is None:
        continue    

    results.append({
        'dataset': dataset,
        'data_type': data_type,
        'model_name': model_name,
        'run_date': metrics_file.parent.name.split('_')[-1],
        'best_epoch': int(best_epoch['epoch']),
        'val_accuracy': float(best_epoch['val_accuracy']),
        'val_f1_score': float(best_epoch['val_f1_score']),
    })
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['dataset', 'data_type', 'val_f1_score'], ascending=False).reset_index(drop=True)
# sort by datset, val f1_score descending
results_df = results_df.sort_values(by=['dataset','val_f1_score'], ascending=False).reset_index(drop=True)
print(results_df)