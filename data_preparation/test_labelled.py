# just to check the distribution of labels in the labelled dataset
import pandas as pd
from pathlib import Path

orig_df0 = pd.read_csv('rosaid/data/rosids23/ROSIDS23.csv')



orig_df = pd.DataFrame()
for csv in Path('rosaid/data/rosids23/').rglob('*.csv'):
    if 'ROSIDS23.csv' in str(csv) or 'labelled_sessions.csv' in str(csv):
        continue
    try:
        df = pd.read_csv(csv)
        atk_name = csv.stem
        def map_label(x):
            if x == "No Label":
                return "NORMAL"
            if int(x) == 1:
                return atk_name
            else:
                return "NORMAL"
        df.Label = df.Label.apply(map_label)
        
        orig_df = pd.concat([orig_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {csv}: {e}")

lbl_df = pd.DataFrame()
images_dir = Path('rosids23_session_images/session_images/')

for csv in Path('rosids23_session_images/').rglob('*.csv'):
    if 'labelled_sessions.csv' in str(csv):
        continue
    try:
        df = pd.read_csv(csv)
        df['file_path'] = df.session_file_name.apply(lambda x: images_dir / x.replace('.pcap','_normalized.png'))
        # ignore non-existing files
        df = df[df.file_path.apply(lambda x: Path(x).exists())]
        lbl_df = pd.concat([lbl_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {csv}: {e}")

# Create mapping dictionary from original labels to labelled data labels
label_mapping = {
    'DoS': 'DoS',
    'NORMAL': 'NORMAL',
    'subscriberflood': 'subcriberflood',
    'unauthorizedpublisher': 'unauthorizedpublisher',
    'unauthorizedsubscriber': 'unauthorizedsubsriber'
}

print(f"Original Label: {orig_df['Label'].value_counts()}")
print(f"Labelled Data: {lbl_df['flow_label'].value_counts()}")
print(f"ROSIDS23 Data:\n{orig_df0['Label'].value_counts()}\n")
print(f"\nTotal Original Records: {len(orig_df)}")
print(f"Total Labelled Records: {len(lbl_df)}")
print(f"Total ROSIDS23 Records: {len(orig_df0)}")
print("\nLabel Mapping Dictionary:")
print(label_mapping)

# Create comparison DataFrame
orig_counts = orig_df['Label'].value_counts()
labelled_counts = lbl_df['flow_label'].value_counts()

comparison_data = []
for orig_label, mapped_label in label_mapping.items():
    orig_count = orig_counts.get(orig_label, 0)
    labelled_count = labelled_counts.get(mapped_label, 0)
    labelled_pct = (labelled_count / orig_count * 100) if orig_count > 0 else 0
    
    comparison_data.append({
        'label': orig_label,
        'orig_count': orig_count,
        'labelled_count': labelled_count,
        'labelled_pct': round(labelled_pct, 2)
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nComparison DataFrame:")
print(comparison_df)

# save labelled df by fixing the labels
# 'BENIGN', 'DOS', 'SUBFLOOD', 'UNAUTHPUB', 'UNAUTHSUB'
final_mapping = {"NORMAL":"BENIGN", "DoS":"DOS", "subcriberflood":"SUBFLOOD", "unauthorizedpublisher":"UNAUTHPUB", "unauthorizedsubsriber":"UNAUTHSUB"}
lbl_df_fixed = lbl_df.copy()
lbl_df_fixed['flow_label'] = lbl_df_fixed['flow_label'].map(final_mapping).fillna(lbl_df_fixed['flow_label'])
lbl_df_fixed.to_csv('rosaid/data/rosids23/labelled_sessions.csv', index=False)
print(f"\nLabelled Data with Fixed Labels: {lbl_df_fixed['flow_label'].value_counts()}")