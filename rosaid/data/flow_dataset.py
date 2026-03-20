import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset as TorchDataset

from rosaid.core.configs import DataSetConfig
from rosaid.core.defs import DataType, NormalizationMethod, SamplingMethod


class DFDataSet:
    def __init__(self, config: DataSetConfig = DataSetConfig()):
        """
        Usage:

        ```python
            train_dataset, val_dataset = DFDataSet(
            config=DataSetConfig(sampling_method=SamplingMethod.UNDERSAMPLE, max_data=100)
        ).get_datasets()

        Args:
            config (DataSetConfig): Configuration for the dataset.

        """
        self.config = config
        self.data_type = DataType.TRAIN
        self.data = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.config.csv_path, low_memory=False)
        # shuffle the data
        df = df.sample(frac=1, random_state=self.config.random_state).reset_index(
            drop=True
        )
        # drop rows with NaN or inf values
        df = df[~np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)]
        df.columns = [c.strip() for c in df.columns]
        # upper labels
        df[self.config.label_column] = df[self.config.label_column].str.upper()
        available_labels = df[self.config.label_column].unique().tolist()
        available_labels = sorted(available_labels)

        if self.config.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.config.label_column}' not found in the data."
            )
        if not self.config.labels:
            self.config.labels = available_labels
        else:
            # filter the data to only include the specified labels
            df = df[df[self.config.label_column].isin(self.config.labels)]
            available_labels = self.config.labels

        if self.config.combine_attacks:
            df[self.config.label_column] = df[self.config.label_column].apply(
                lambda x: (
                    self.config.normal_label
                    if x == self.config.normal_label
                    else "Attack"
                )
            )
            available_labels = [self.config.normal_label, "Attack"]

        # select max data by class
        if self.config.max_data > 0:
            # if max_data is float, find the max number of samples per class
            lbl_cnts = df[self.config.label_column].value_counts().to_dict()
            if isinstance(self.config.max_data, float):
                nlbl_cnts = {
                    k: int(v * self.config.max_data) for k, v in lbl_cnts.items()
                }
            else:
                nlbl_cnts = {k: self.config.max_data for k, v in lbl_cnts.items()}

            logger.info(f"Limiting data to {nlbl_cnts}.")
            df = (
                df.groupby(self.config.label_column)
                .apply(
                    lambda x: x.sample(
                        min(len(x), nlbl_cnts[x.name]),
                        replace=False,
                        random_state=self.config.random_state,
                    )
                )
                .reset_index(drop=True)
            )

        self.label_encoding = {label: idx for idx, label in enumerate(available_labels)}
        # label encoding to one hot encoding
        for label in self.label_encoding.keys():
            lbl = [0] * len(self.label_encoding)
            idx = self.label_encoding[label]
            lbl[idx] = 1
            self.label_encoding[label] = lbl
        if df.empty:
            raise ValueError("No data found in the specified CSV files.")

        return df

    def get_synthetic_train_val(self, df: pd.DataFrame):
        df["is_synthetc"] = df.is_synthetic.apply(
            lambda x: True if x == "True" or x is True else False
        )
        logger.info(
            f"Data contains synthetic samples: {df['is_synthetic'].value_counts().to_dict()}"
        )
        val_df = pd.DataFrame()
        for label in df[self.config.label_column].unique():
            label_df = df.query(
                f"{self.config.label_column} == '{label}' and is_synthetic != True"
            )
            if not label_df.empty:
                val_df = pd.concat(
                    [
                        val_df,
                        label_df.sample(
                            frac=1 - self.config.train_ratio,
                            random_state=self.config.random_state,
                        ),
                    ]
                )

        train_df = df.drop(val_df.index)
        return train_df, val_df

    def get_datasets(self):
        # only keep the features and label column
        # no need to filter it here
        # self.data = self.data[self.config.features + [self.config.label_column]]

        # split the data into train and validation sets
        if self.config.train_ratio <= 0 or self.config.train_ratio >= 1:
            raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
        if len(self.config.labels) and self.config.combine_attacks is False:
            self.data = self.data[
                self.data[self.config.label_column].isin(self.config.labels)
            ]
        if len(self.config.features) == 0:
            # select all numeric columns except the label column
            discard_cols = [
                "Unnamed: 0",
                "Src Port",
                "Dst Port",
                "Protocol",
                "File",
                "Unnamed: 0",
                "Unnamed: 0.1",
                "firstPacketDIR",
            ]
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in discard_cols]
            if self.config.label_column in numeric_cols:
                numeric_cols.remove(self.config.label_column)
            self.config.features = numeric_cols
            logger.info(
                f"No features specified. Using all columns except the label column: {self.config.features}"
            )

        label_counts = self.data[self.config.label_column].value_counts()
        logger.info(f"Label counts in the dataset: {label_counts.to_dict()}")
        # sample data for training and validation by grouping by label
        self.data["idx"] = self.data.index
        if not self.config.has_synthetic:
            logger.info("Filtering out synthetic data for training and validation.")
            self.data["is_synthetic"] = False
            train_df, validation_df = train_test_split(
                self.data,
                test_size=1 - self.config.train_ratio,
                stratify=self.data[self.config.label_column],
                random_state=self.config.random_state,
            )
        else:
            train_df, validation_df = self.get_synthetic_train_val(self.data)
            logger.info(
                f"Has Synthetic Counts (Train): {train_df['is_synthetic'].value_counts().to_dict()}"
            )
            logger.info(
                f"Has Synthetic Counts (Val): {validation_df['is_synthetic'].value_counts().to_dict()}"
            )

        logger.info(
            f"Training dataset sampled with label counts: {train_df[self.config.label_column].value_counts().to_dict()}"
        )

        logger.info(
            f"Validation dataset sampled with label counts: {validation_df[self.config.label_column].value_counts().to_dict()}"
        )

        # apply over/under sampling on trainset only
        if self.config.sampling_method == SamplingMethod.OVERSAMPLE:
            logger.info("Applying SMOTE oversampling to the training dataset.")
            # first use random oversampling to increase minority classes by half of needed
            value_counts = train_df[self.config.label_column].value_counts()
            max_count = value_counts.max()
            target_counts = (max_count // 2) + 1
            for label, count in value_counts.items():
                if count < target_counts:
                    n_samples = target_counts - count
                    sampled_df = train_df[
                        train_df[self.config.label_column] == label
                    ].sample(
                        n_samples,
                        replace=True,
                        random_state=self.config.random_state,
                    )
                    train_df = pd.concat([train_df, sampled_df])
            logger.info(
                f"Training dataset size after random oversampling: {train_df.shape}"
            )
            # reset index after oversampling
            train_df = train_df.reset_index(drop=True)

            smote = SMOTE(random_state=self.config.random_state)
            X_train, y_train = smote.fit_resample(
                train_df[self.config.features], train_df[self.config.label_column]
            )
            train_df = pd.DataFrame(X_train, columns=self.config.features)
            train_df[self.config.label_column] = y_train
        elif self.config.sampling_method == SamplingMethod.UNDERSAMPLE:
            logger.info("Applying RandomUnderSampler to the training dataset.")
            undersampler = RandomUnderSampler(random_state=self.config.random_state)
            X_train, y_train = undersampler.fit_resample(
                train_df[self.config.features], train_df[self.config.label_column]
            )
            train_df = pd.DataFrame(X_train, columns=self.config.features)
            train_df[self.config.label_column] = y_train
        else:
            logger.info("No sampling applied to the training dataset.")
        # shuffle again after sampling
        train_df = train_df.sample(
            frac=1, random_state=self.config.random_state
        ).reset_index(drop=True)
        # apply normalization on train and validation datasets
        if self.config.normalization_method == NormalizationMethod.MIN_MAX:
            logger.info("Applying Min-Max normalization to the datasets.")
            scaler = MinMaxScaler()
            logger.info(f"Features:{self.config.features}")
            logger.info(f"Columns: {train_df.columns}")
            train_df[self.config.features] = scaler.fit_transform(
                train_df[self.config.features]
            )
        elif self.config.normalization_method == NormalizationMethod.STANDARD:
            logger.info("Applying Standard normalization to the datasets.")
            scaler = StandardScaler()
            train_df[self.config.features] = scaler.fit_transform(
                train_df[self.config.features]
            )
        else:
            logger.info("No normalization applied to the datasets.")
            scaler = None

        if scaler is not None:
            validation_df[self.config.features] = scaler.transform(
                validation_df[self.config.features]
            )
        self.scaler = scaler

        train_dataset = DFDataSet(
            config=self.config,
        )
        validation_dataset = DFDataSet(
            config=self.config,
        )

        train_df = train_df.reset_index(drop=True)
        validation_df = validation_df.reset_index(drop=True)

        train_dataset.data = train_df
        validation_dataset.data = validation_df
        train_dataset.data_type = DataType.TRAIN
        validation_dataset.data_type = DataType.VALIDATION
        train_dataset.label_encoding = self.label_encoding
        validation_dataset.label_encoding = self.label_encoding
        train_dataset.scaler = self.scaler
        validation_dataset.scaler = self.scaler

        logger.info(
            f"Train dataset size: {train_dataset.data.shape}, Validation dataset size: {validation_dataset.data.shape}"
        )

        return train_dataset, validation_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        X = row[self.config.features].values.astype(float)
        y = row[self.config.label_column]
        if y not in self.label_encoding:
            raise ValueError(f"Label '{y}' not found in class encoding.")
        y = self.label_encoding[y]

        return X, y


class CLFDataSet(TorchDataset):
    def __init__(self, df_ds: DFDataSet):
        self.dataset = df_ds
        self.config = df_ds.config
        self.data = df_ds.data
        self.label_encoding = df_ds.label_encoding
        self.data_type = df_ds.data_type
        self.num_classes = len(self.label_encoding)
        # Get class counts
        class_counts = self.data[self.config.label_column].value_counts().to_dict()

        # Compute inverse frequency weights
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}

        # Order weights according to label encoding
        weights_list = [class_weights[label] for label in self.label_encoding.keys()]

        # Convert to tensor (no need to normalize)
        self.class_weights = torch.tensor(weights_list, dtype=torch.float32)
        self.class_weights = (
            self.class_weights * len(self.class_weights) / self.class_weights.sum()
        )
        self.label_counts = class_counts
        self.X = self.data[self.config.features].values.astype(float)
        self.y = self.data[self.config.label_column].values
        logger.info(f"Label encoding: {self.label_encoding}")
        self.batch_size = self.num_classes * 10
        self.curr_idx = 0
        self.label_batch_counts = {lbl: 0 for lbl in self.label_encoding.keys()}
        self.curr_X_batch = []
        self.curr_y_batch = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == DataType.TRAIN:
            # prepare a batch of data with balanced classes
            # and return the next item in the batch
            if self.curr_idx == 0:
                # for each label, sample a data
                num_samples = self.batch_size // self.num_classes
                self.curr_X_batch = []
                self.curr_y_batch = []
                for label, idxs in self.label_encoding.items():
                    label_idxs = np.where(self.y == label)[0]
                    if len(label_idxs) < num_samples:
                        Warning(
                            f"Not enough samples for label '{label}' to create a balanced batch."
                        )
                    sampled_idxs = np.random.choice(
                        label_idxs, num_samples, replace=True
                    )
                    self.curr_X_batch.extend(self.X[sampled_idxs])
                    self.curr_y_batch.extend([label] * num_samples)
                self.curr_idx = len(self.curr_X_batch)

            # randomly select an index from the current batch
            idx = np.random.randint(0, len(self.curr_X_batch))
            X = self.curr_X_batch[idx]
            y = self.curr_y_batch[idx]
            self.curr_idx -= 1

            self.label_batch_counts[y] += 1
            if self.curr_idx == 0:
                # logger.info(f"Batch completed with counts: {self.label_batch_counts}")
                self.label_batch_counts = {
                    lbl: 0 for lbl in self.label_batch_counts.keys()
                }

        else:
            X = self.X[idx]
            y = self.y[idx]
        if y not in self.label_encoding:
            raise ValueError(f"Label '{y}' not found in class encoding.")
        lbl_str = y
        y = self.label_encoding[y]
        # convert to tensor
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        X = torch.from_numpy(X)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


if __name__ == "__main__":
    config = DataSetConfig()
    dataset1 = DFDataSet(config=config)
    train_ds, val_ds = dataset1.get_datasets()

    dataset2 = DFDataSet(config=config)
    train_ds2, val_ds2 = dataset2.get_datasets()
    print("Train Dataset Length:", len(train_ds2))
    print("Validation Dataset Length:", len(val_ds2))
