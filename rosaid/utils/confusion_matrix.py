import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics.functional.classification import multiclass_f1_score, binary_f1_score
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path


def get_confusion_matrix(
    predictions,
    targets,
    label_keys,
    out_file: Path | None = None,
    eps=0,
):
    if len(label_keys)>2:

        f1 = multiclass_f1_score(
            torch.tensor(predictions),
            torch.tensor(targets),
            num_classes=len(label_keys),
            average="macro",
        )
    else:
        f1 = binary_f1_score(
            torch.tensor(predictions),
            torch.tensor(targets),
            
        )

    cm = confusion_matrix(targets, predictions)

    if out_file is not None:

        plt.figure(figsize=(12, 10))  # Increased width for better label spacing

        # Create heatmap
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_keys,
            yticklabels=label_keys,
        )
        title = "Confusion Matrix: {:.4f} F1 Score".format(f1.item())
        if eps > 0:
            title += f" | eps={eps}"
        ax.set_title(title, fontsize=14)

        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title(title, fontsize=14)

        # Rotate and align x-axis labels
        plt.xticks(rotation=45, ha="right", fontsize=10)

        # Adjust layout to prevent clipping
        plt.tight_layout(pad=3.0)  # Add padding around plot

        # Save with tight bounding box
        plt.savefig(
            out_file,
            bbox_inches="tight",  # Ensures all elements are included
        )
        logger.info(f"Confusion matrix saved to {out_file}")
        plt.close()
    return f1.item(), cm
