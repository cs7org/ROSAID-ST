import pandas as pd
from pathlib import Path

from loguru import logger

import shutil

data_src = Path(
    r"data\dnp3data\DNP3_Intrusion_Detection_Dataset_Final"
)
data_dest = Path(r"data\Custom_DNP3_Parser")

for file in data_src.rglob("*.csv"):
    try:
        logger.info(f"Processing {file.name}")
        if "CIC" in str(file):
            logger.info(f"Skipping {file.name}, contains 'CIC'")
            continue
        if "Balanced" in str(file):
            logger.info(f"Skipping {file.name}, contains 'Balanced'")
            continue
        # Determine the timeout folder based on the filename
        if "45_timeout" in str(file):
            timeout_folder = "45_timeout"
        elif "60_timeout" in str(file):
            timeout_folder = "60_timeout"
        elif "75_timeout" in str(file):
            timeout_folder = "75_timeout"
        elif "120_timeout" in str(file):
            timeout_folder = "120_timeout"
        elif "240_timeout" in str(file):
            timeout_folder = "240_timeout"
        else:
            logger.warning(f"Skipping {file.name}, no matching timeout folder found.")
            continue

        # Create the destination folder if it doesn't exist
        dest_folder = data_dest / timeout_folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Copy the file to the destination folder
        shutil.copy(file, dest_folder)
        logger.info(f"Copied {file.name} to {dest_folder}")
    except Exception as e:
        logger.error(f"Error processing {file}: {e}")

data_shape = {}
data_df = {}

for timeout in ["45_timeout", "60_timeout", "75_timeout", "120_timeout", "240_timeout"]:
    timeout_folder = data_dest / timeout
    combined_df = pd.DataFrame()
    for file in timeout_folder.rglob("*.csv"):
        try:
            df = pd.read_csv(file, low_memory=False)
            # Combine the dataframes
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    data_shape[timeout] = combined_df.shape
    data_df[timeout] = combined_df
    logger.info(f"Combined shape for {timeout}: {data_shape[timeout]}")

for timeout, df in data_df.items():
    if "Label" not in df.columns:
        logger.warning(f"Label column not found in {timeout} DataFrame. Skipping.")
        df["Label"] = df[" Label"]
        del df[" Label"]

    # Save the combined DataFrame to a CSV file
    output_file = data_dest / f"combined_{timeout}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved combined DataFrame for {timeout} to {output_file}")
