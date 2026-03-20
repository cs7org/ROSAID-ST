from pathlib import Path

from pydantic import BaseModel, Field

from rosaid.core.defs import (
    TOP_CIC_FEATURES,
    MetricType,
    NormalizationMethod,
    Optimizer,
    SamplingMethod,
)


class DataSetConfig(BaseModel):
    csv_path: Path = Field(
        default=Path(
            r"E:\MSc Works\IDS\data\Custom_DNP3_Parser\combined_45_timeout.csv"
        ),
        description="Path to the directory containing CSV files.",
    )
    label_column: str = Field(
        default="Label",
        description="Column name in the CSV files that contains the labels.",
    )
    labels: list[str] = Field(
        default=[
            "REPLAY",
            "DNP3_INFO",
            "DNP3_ENUMERATE",
            "STOP_APP",
            "NORMAL",
            "INIT_DATA",
            "COLD_RESTART",
            "WARM_RESTART",
            "DISABLE_UNSOLICITED",
        ],
        description="List of labels to be used for classification.",
    )
    normal_label: str = Field(
        default="NORMAL",
        description="Label for normal data samples.",
    )
    combine_attacks: bool = Field(
        default=False,
        description="Whether to combine all attack types into a single label.",
    )
    max_data: int | float = Field(
        default=-1,
        description="Maximum number of samples to be used from each class.",
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.NONE,
        description="Method to handle class imbalance.",
    )
    train_ratio: float = Field(
        default=0.75,
        description="Ratio of the dataset to be used for training.",
    )

    features: list[str] = Field(
        default=TOP_CIC_FEATURES,
        description="List of feature names to be used for training. Default is selected from PFI.",
    )
    normalization_method: NormalizationMethod = Field(
        default=NormalizationMethod.MIN_MAX,
        description="Method to normalize the data.",
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility.",
    )
    has_synthetic: bool = Field(
        default=False,
        description="Whether the dataset contains synthetic data.",
    )


class NNTrainerConfig(BaseModel):
    result_dir: Path
    expt_name: str
    run_name: str
    log_every: int = 1
    chkpt_every: int = 0
    best_model_name: str = "best_model.pth"
    best_model_metric: MetricType = MetricType.LOSS
    best_model_metric_greater: bool = False
    optimizer: Optimizer = Optimizer.ADAM
    device: str = "cuda"
    epochs: int = 100
    batch_size: int = 32
    shuffle: bool = True
    number_of_workers: int = 0
    log_file: str = "trainer.log"
    metric_file: str = "metrics.csv"
    metrics: list[callable] = Field(
        default_factory=lambda: [
            MetricType.ACCURACY,
            MetricType.PRECISION,
            MetricType.RECALL,
            MetricType.F1_SCORE,
        ],
        description="List of metrics to be computed during training.",
    )
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    early_stopping_patience: int = 50
    weighted_loss: bool = True
    log_mlflow: bool = True
    lr_scheduler: str = "ReduceLROnPlateau"
    random_seed: int = 42
    is_binary_classification: bool = False
    run_dir: Path|None = None



    class Config:
        arbitrary_types_allowed = True
