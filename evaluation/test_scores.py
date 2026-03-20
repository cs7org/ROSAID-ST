import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# df = pd.read_csv('rosaid/results/evaluation/ROSIDS23normalized_frequency/blockcnn2d_nosampling_binary_20260123/detailed_results_blockcnn2d_nosampling_binary_20260123.csv')

# true_value = df['true_value'].values
# pred_value = df['pred_score'].values
# accuracy = accuracy_score(true_value, pred_value)
# macro_f1 = f1_score(true_value, pred_value, average='macro')
# micro_f1 = f1_score(true_value, pred_value, average='micro')

# print(f"Accuracy: {accuracy}")
# print(f"Macro F1 Score: {macro_f1}")
# print(f"Micro F1 Score: {micro_f1}")

eval_root = Path('rosaid/results/evaluation')
valid_models = ['ROSIDS23_ORIGINAL', 'ROSIDS23_FILTERED', 'ROSIDS23_NORMALIZED',
                    'ROSIDS23normalized_frequency',
                    'image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260102']

# model_name, data_type, accuracy, macro_f1, micro_f1
columns = ['model_name', 'data_type', 'accuracy', 'macro_f1', 
           'micro_f1', 'model_name_full', 'run_date', 'filtered_columns']
result = []
for model in valid_models:
    model_path = eval_root / model
    detailed_results = list(model_path.rglob('detailed_results_*.csv'))
    for res_file in detailed_results:
        model_name = 'BlockCNN2D' if 'blockcnn2d' in res_file.name else 'MobileNetV3Large' if 'mobilenet' in res_file.name else 'ResNet18'
        run_date = res_file.parent.name.split('_')[-1]
        df = pd.read_csv(res_file)
        model_name_full = res_file.parent.name
        true_value = df['true_value'].values
        pred_value = df['pred_score'].values
        accuracy = accuracy_score(true_value, pred_value)
        macro_f1 = f1_score(true_value, pred_value, average='macro')
        micro_f1 = f1_score(true_value, pred_value, average='micro')
        col_filtered= False
        if 'filtered_columns' in df.columns:
            col_filtered = df['filtered_columns'].values[0]

        result.append([model_name,model,
            accuracy, macro_f1, micro_f1, model_name_full, run_date, col_filtered
        ])

result_df = pd.DataFrame(result, columns=columns)
# sort descending by macro_f1
result_df = result_df.sort_values(by='macro_f1', ascending=False).reset_index(drop=True)
print(result_df)
result_df.to_csv(eval_root / 'summary_model_evaluation.csv', index=False)