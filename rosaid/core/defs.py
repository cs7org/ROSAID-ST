from dataclasses import dataclass
from enum import Enum

import pandas as pd


class NormalImageType(int, Enum):
    ORIGINAL = 0
    FILTERED = 1
    NORMALIZED = 2
    FILTERED_GRAM = 3
    NORMALIZED_GRAM = 4
    ZSCORE = 5
    ZGRAM1D = 6
    ZGRAM3D = 7
    UNFILTERED_GRAM = 8
    UNFILTERED_GRAM3D = 9
    FILTERED_GRAM3D = 10


@dataclass
class Session:
    flow_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    packets: list["Packet"]
    interval: float
    raw_bytes: list[bytearray]
    label: str = "NORMAL"
    num_forward_packets: int = 0
    num_backward_packets: int = 0
    expected_forward_packets: int = 0
    expected_backward_packets: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    def __repr__(self):
        return (
            f"Session(start_time={self.start_time}, end_time={self.end_time}, "
            f"num_packets={self.num_packets}, interval={self.interval})"
        )


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"


class DataType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"

    def __str__(self) -> str:
        return self.value


class SamplingMethod(str, Enum):
    OVERSAMPLE = "oversampling"
    UNDERSAMPLE = "undersampling"
    NONE = "nosampling"


class NormalizationMethod(str, Enum):
    MIN_MAX = "min_max"
    STANDARD = "standard"


class MetricType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LOSS = "loss"


TOP_CIC_FEATURES = [
    "Subflow Bwd Pkts",
    "Bwd Header Len",
    "Tot Bwd Pkts",
    "Tot Fwd Pkts",
    "Fwd IAT Mean",
    "Fwd Pkt Len Min",
    "Subflow Fwd Byts",
    "Fwd Header Len",
    "TotLen Bwd Pkts",
    "Subflow Bwd Byts",
    "Fwd Act Data Pkts",
    "Bwd Pkt Len Mean",
    "Flow IAT Std",
    "Pkt Len Var",
    "Fwd Seg Size Avg",
    "Flow Byts/s",
    "TotLen Fwd Pkts",
    "Pkt Len Mean",
    "Flow Pkts/s",
    "Pkt Len Std",
    "Fwd Pkts/s",
    "Bwd Pkt Len Std",
    "Fwd Pkt Len Mean",
    "Pkt Len Max",
    "Bwd IAT Mean",
    "Bwd Pkt Len Max",
    "Flow IAT Mean",
    "Pkt Size Avg",
    "Subflow Fwd Pkts",
    "Bwd Seg Size Avg",
    "Dst Port",
    "Bwd IAT Tot",
    "Fwd IAT Std",
    "Bwd IAT Std",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Fwd Pkt Len Std",
    "Init Bwd Win Byts",
    "Bwd Pkt Len Min",
    "Fwd Pkt Len Max",
    "Flow Duration",
    "Src Port",
    "ACK Flag Cnt",
    "Fwd IAT Tot",
    "Bwd IAT Max",
    "Bwd PSH Flags",
    "PSH Flag Cnt",
    "Idle Mean",
    "Active Max",
    "Protocol",
    "Fwd IAT Max",
    "Flow IAT Max",
    "Down/Up Ratio",
    "Idle Max",
    "Bwd IAT Min",
    "Active Min",
    "Fwd IAT Min",
    "Active Std",
    "Active Mean",
    "Init Fwd Win Byts",
    "Bwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "FIN Flag Cnt",
    "Bwd URG Flags",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Seg Size Min",
    "Fwd Blk Rate Avg",
    "Fwd Pkts/b Avg",
    "Fwd Byts/b Avg",
    "URG Flag Cnt",
    "Flow IAT Min",
    "Idle Std",
    "Idle Min",
]

IEC104_TOP_FEATURES = [
    "Packet Length Mean",
    "Packet Length Max",
    "Flow IAT Min",
    "Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Min",
    "SYN Flag Count",
    "Fwd Packet Length Min",
    "Bwd Segment Size Avg",
    "Average Packet Size",
    "Bwd Packet Length Max",
    "Fwd PSH Flags",
    "Total Length of Fwd Packet",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Bwd IAT Min",
    "Total Length of Bwd Packet",
    "Packet Length Variance",
    "FWD Init Win Bytes",
    "PSH Flag Count",
    "ACK Flag Count",
    "Fwd IAT Min",
    "Bwd Init Win Bytes",
    "Fwd Segment Size Avg",
    "Protocol",
    "RST Flag Count",
    "Packet Length Std",
    "Fwd Packet Length Max",
    "Bwd IAT Std",
    "FIN Flag Count",
    "Bwd IAT Max",
    "Bwd Packet Length Std",
    "Fwd Seg Size Min",
    "Bwd Bulk Rate Avg",
    "Fwd Header Length",
    "Bwd Header Length",
    "Dst Port",
    "Fwd IAT Std",
    "Fwd Packets/s",
    "Fwd Packet Length Std",
    "Fwd IAT Max",
    "Bwd IAT Total",
    "Flow IAT Max",
    "Total Bwd packets",
    "Bwd Bytes/Bulk Avg",
    "Fwd IAT Mean",
    "Src Port",
    "Flow IAT Mean",
    "Fwd Act Data Pkts",
    "Bwd Packet/Bulk Avg",
    "Fwd IAT Total",
    "Flow IAT Std",
    "Bwd IAT Mean",
    "Total Fwd Packet",
    "Flow Packets/s",
    "Active Std",
    "Idle Std",
    "Subflow Bwd Packets",
    "Active Max",
    "Subflow Fwd Bytes",
    "Subflow Fwd Packets",
    "Subflow Bwd Bytes",
    "Fwd Bytes/Bulk Avg",
    "Down/Up Ratio",
    "Idle Mean",
    "Active Mean",
    "Active Min",
    "Bwd PSH Flags",
    "Bwd Packets/s",
    "Bwd URG Flags",
    "CWR Flag Count",
    "URG Flag Count",
    "ECE Flag Count",
    "Flow Duration",
    "Fwd Bulk Rate Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd URG Flags",
    "Idle Max",
    "Idle Min",
]

TOP_ROSIDS23_FEATURES = [
    "Fwd IAT Min",
    "Fwd IAT Mean",
    "Src Port",
    "Bwd Header Len",
    "Bwd IAT Max",
    "Dst Port",
    "Subflow Bwd Pkts",
    "Fwd IAT Max",
    "Bwd IAT Mean",
    "Tot Bwd Pkts",
    "Flow IAT Mean",
    "Bwd IAT Tot",
    "Bwd IAT Std",
    "Tot Fwd Pkts",
    "Fwd IAT Std",
    "Flow IAT Max",
    "Pkt Len Max",
    "Fwd Header Len",
    "Flow Pkts/s",
    "Fwd IAT Tot",
    "Bwd Pkts/s",
    "Init Bwd Win Byts",
    "Flow Duration",
    "Pkt Len Mean",
    "Fwd Pkts/s",
    "Active Min",
    "Active Std",
    "Pkt Len Var",
    "Bwd Seg Size Avg",
    "Flow Byts/s",
    "Active Mean",
    "Idle Mean",
    "Bwd Pkt Len Max",
    "Flow IAT Std",
    "ACK Flag Cnt",
    "TotLen Fwd Pkts",
    "Fwd Pkt Len Std",
    "Fwd Seg Size Avg",
    "Subflow Bwd Byts",
    "Pkt Len Std",
    "Idle Max",
    "Idle Std",
    "Pkt Len Min",
    "RST Flag Cnt",
    "Bwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Init Fwd Win Byts",
    "Fwd Blk Rate Avg",
    "Bwd Pkts/b Avg",
    "Fwd URG Flags",
    "Fwd Seg Size Min",
    "URG Flag Cnt",
    "Bwd URG Flags",
    "ECE Flag Cnt",
    "Fwd Byts/b Avg",
    "Fwd PSH Flags",
    "CWE Flag Count",
    "Fwd Pkts/b Avg",
    "Active Max",
    "Bwd Pkt Len Min",
    "TotLen Bwd Pkts",
    "Idle Min",
    "Bwd PSH Flags",
    "PSH Flag Cnt",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Subflow Fwd Pkts",
    "Fwd Act Data Pkts",
    "Protocol",
    "Bwd Pkt Len Std",
    "Bwd Pkt Len Mean",
    "Fwd Pkt Len Max",
    "Pkt Size Avg",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "Subflow Fwd Byts",
    "Bwd IAT Min",
    "Down/Up Ratio",
    "Flow IAT Min",
]

BEST_CIC_FEATURE_INDEX = 43
BEST_IEC_FEATURES_INDEX = 20
BEST_ROSIDS23_FEATURES_INDEX = 8

# NOTE: Make sure first label is normal
DNP3_CLASSES: list[str] = [
    "NORMAL",
    "REPLAY",
    "DNP3_INFO",
    "DNP3_ENUMERATE",
    "STOP_APP",
    "INIT_DATA",
    "COLD_RESTART",
    "WARM_RESTART",
    "DISABLE_UNSOLICITED",
]

# MITM is not included
IEC104_CLASSES: list[str] = [
    "ATTACKFREE",
    "DOSATTACK",
    "FLOODATTACK",
    "FUZZYATTACK",
    "IEC104STARVATIONATTACK",
    "NTPDDOSATTACK",
    "PORTSCANATTACK",
]

# ROSIDS23_CLASSES includes only attack types present in the dataset
ROSIDS23_CLASSES: list[str] = ["BENIGN", "DOS", "SUBFLOOD", "UNAUTHPUB", "UNAUTHSUB"]
