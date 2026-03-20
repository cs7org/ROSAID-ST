import sys
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from torch import nn
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from rosaid.core.configs import NNTrainerConfig, Optimizer
from rosaid.data.dataset import TorchImageDataset


class NNTrainer:
    def __init__(
        self,
        config: NNTrainerConfig,
        model: nn.Module,
        train_dataset: TorchImageDataset,
        val_dataset: TorchImageDataset,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        min_lr: float = 1e-7,
    ):
        self.model = model
        self.config = config
        if self.config.run_dir is None:
            self.config.run_dir = (
                self.config.result_dir
                / self.config.expt_name
                / self.config.run_name
            )
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_file != "":
            if (self.config.run_dir / self.config.log_file).exists():
                (self.config.run_dir / self.config.log_file).unlink()
            logger.info(f"Logging to {self.config.run_dir / self.config.log_file}")
            logger.add(
                self.config.run_dir / self.config.log_file,
                rotation="1 day",
                retention="7 days",
                level="INFO",
                format="{time} | {level} | {message}",
            )
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        logger.info(f"Model attribs: {self.model.__dict__}")

        if self.config.optimizer == Optimizer.ADAM:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif self.config.optimizer == Optimizer.SGD:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        elif self.config.optimizer == Optimizer.ADAMW:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        self.criterion = criterion
        self._metrics = config.metrics
        self.metric_history = defaultdict(list)
        self.patience_counter = 0
        self.started_mlflow = False
        self.epoch = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.9,
            patience=5,
            min_lr=min_lr,
        )
        self.random_state = np.random.RandomState(self.config.random_seed)

        num_classes = train_dataset.num_classes
        batch_size = config.batch_size
        if batch_size % num_classes != 0:
            logger.warning(
                f"Batch size {batch_size} is not divisible by number of classes {num_classes}. "
                "This may lead to imbalanced batches."
            )
            # increase it to the next multiple of num_classes
            config.batch_size = (batch_size // num_classes + 1) * num_classes

        train_dataset.batch_size = config.batch_size

        # write scaler if it exists
        if hasattr(train_dataset.dataset, "scaler") and train_dataset.dataset.scaler:
            logger.info("Saving scaler to disk.")
            joblib.dump(
                train_dataset.dataset.scaler,
                self.config.run_dir / "scaler.pkl",
            )

        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Configuration: {self.config.model_dump_json(indent=2)}")
        logger.info(f"Data Config: {train_dataset.config.model_dump_json(indent=2)}")
        logger.info(
            f"Training dataset: {len(train_dataset)} samples, "
            f"Validation dataset: {len(val_dataset)} samples."
        )
        try:
            lbl_col = train_dataset.config.label_column
            logger.info(
                f"Train Labels: {train_dataset.data[lbl_col].value_counts().to_dict()}"
            )
            logger.info(
                f"Validation Labels: {val_dataset.data[lbl_col].value_counts().to_dict()}"
            )
        except Exception as e:
            logger.warning(f"Could not log label counts: {e}")
        # save config
        with open(self.config.run_dir / "train_config.json", "w") as f:
            f.write(self.config.model_dump_json(indent=2))
        # save dataset config
        with open(self.config.run_dir / "dataset_config.json", "w") as f:
            f.write(train_dataset.config.model_dump_json(indent=2))
        self.logger = logger
        self.logger.info("Trainer initialized successfully.")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # loaders
        self.make_dataloaders()
        self.metrics = self.get_metrics()
        if self.config.weighted_loss:
            self.update_criterion(self.criterion)

    def make_dataloaders(self):
        """Create dataloaders for training and validation datasets."""
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=min(8, self.config.number_of_workers),
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(8, self.config.number_of_workers),
        )

    def update_criterion(self, criterion: nn.Module):
        """Update the loss criterion."""
        if isinstance(criterion, nn.CrossEntropyLoss):
            self.criterion = nn.CrossEntropyLoss(
                weight=self.train_dataset.class_weights.to(self.device)
            )
            logger.info(f"Label Counts: {self.train_dataset.label_counts}")
            labels = list(self.train_dataset.label_encoding.keys())
            class_wts = self.train_dataset.class_weights.tolist()
            lbl_wts = {k: v for k, v in zip(labels, class_wts)}
            logger.info(
                f"Criterion updated to CrossEntropyLoss with class weights: {lbl_wts}."
            )

        else:
            logger.warning(
                "Criterion is not CrossEntropyLoss. Using the provided criterion."
            )

    def get_metrics(self):
        """Get the metrics based on the configuration."""
        num_classes = self.train_dataset.num_classes

        metrcs = {}
        for metric in self.config.metrics:
            if metric == "accuracy":
                metrcs["accuracy"] = Accuracy(
                    task="binary" if num_classes < 2 else "multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
            elif metric == "precision":
                metrcs["precision"] = Precision(
                    task="binary" if num_classes < 2 else "multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
            elif metric == "recall":
                metrcs["recall"] = Recall(
                    task="binary" if num_classes < 2 else "multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
            elif metric == "f1_score":
                metrcs["f1_score"] = F1Score(
                    task="binary" if num_classes < 2 else "multiclass",
                    num_classes=num_classes,
                    average="macro",
                )

        metrcs = MetricCollection(metrcs)
        metrics = metrcs.to(self.device)
        logger.info(f"Metrics initialized: {list(metrics.keys())}")
        return metrics

    def forward_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        if self.config.is_binary_classification and labels.ndim == 2:
            # convert to single label
            labels = labels.argmax(dim=1)
            logits, probs = self.model(inputs)
            # prepare weights based on labels
            weights = []
            for lbl in labels:
                weights.append(self.train_dataset.class_weights[lbl.item()].item())
            loss = nn.functional.binary_cross_entropy(
                probs.flatten(),
                labels.float(),
                weight=torch.tensor(weights, device=self.device),
            )
            # convert probs to two class probs
            probs = probs >= 0.5
            probs = torch.cat((1 - probs.float(), probs.float()), dim=1)
            # loss = self.criterion(probs.flatten(), labels.float())
        else:
            logits, probs = self.model(inputs)
            loss = self.criterion(logits, labels)

        # Update metrics
        self.metrics.update(
            probs.argmax(dim=1), labels.argmax(dim=1) if labels.ndim == 2 else labels
        )
        metrics = self.metrics.compute()
        # itemize
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        return probs, loss, metrics

    def at_batch_end(self):
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        pass

    def at_epoch_end(self):
        pass

    def run_epoch(self, dataloader, is_train=True):
        self.is_train = is_train
        epoch_loss = 0.0
        epoch_metrics = {metric: 0.0 for metric in self.metrics}
        self.metrics.reset()
        epoch_metrics["loss"] = 0.0
        pbar = tqdm(
            dataloader,
            desc="Training" if is_train else "Validation",
            unit="batch",
            disable=not sys.stdout.isatty(),  # Disable tqdm in non-interactive environments
        )
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for i, batch in enumerate(pbar):
            if is_train:
                self.optimizer.zero_grad()
            outputs, loss, metrics = self.forward_step(batch)
            if is_train:
                loss.backward()
                self.at_batch_end()
                self.optimizer.step()
            # if any metric in metrics is not in self.metrics, update self.metrics
            for metric, value in metrics.items():
                if metric not in epoch_metrics:
                    epoch_metrics[metric] = 0.0
                epoch_metrics[metric] += value

            # add loss as first metric
            epoch_metrics["loss"] += loss.item()
            if is_train:
                pbar.set_description(
                    f"Epoch[{self.epoch}/{self.config.epochs}] - Training"
                )
            else:
                pbar.set_description(
                    f"Epoch[{self.epoch}/{self.config.epochs}] - Validation"
                )

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    **{
                        metric: epoch_metrics[metric] / (i + 1)
                        for metric in epoch_metrics.keys()
                    },
                }
            )
        epoch_loss /= len(dataloader)
        for metric in metrics:
            epoch_metrics[metric] /= len(dataloader)
        epoch_metrics["loss"] = epoch_loss
        return outputs, epoch_loss, epoch_metrics

    def train(self):
        logger.info("Starting training process...")
        if self.config.log_mlflow:
            # if mlflow object is not None, log metrics to mlflow
            if not self.started_mlflow:
                try:
                    mlflow.set_tracking_uri("http://localhost:5000")
                    mlflow.set_experiment(self.config.expt_name)
                    mlflow.start_run(
                        run_name=self.config.run_name,
                        nested=False,
                    )
                    logger.info("MLflow run started.")
                    self.started_mlflow = True
                except Exception as e:
                    logger.error(f"Failed to start MLflow run: {e}")
                    logger.warning(
                        "MLflow is not running. Metrics will not be logged to MLflow."
                    )
                    self.started_mlflow = False
                    self.config.log_mlflow = False

        self.model.train()
        best_metric_value = (
            float("-inf") if self.config.best_model_metric_greater else float("inf")
        )
        best_model_path = self.config.run_dir / self.config.best_model_name

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            # Training step
            outputs, epoch_loss, epoch_metrics = self.run_epoch(
                self.train_loader, is_train=True
            )

            # Validation step
            with torch.no_grad():
                outputs, val_loss, val_metrics = self.run_epoch(
                    self.val_loader, is_train=False
                )
            # scheduler step
            self.scheduler.step(metrics=val_loss)
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.at_epoch_end()
            metrics = val_metrics.keys()
            logger.info(
                f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}, "
                + ", ".join(
                    f"{metric}: {epoch_metrics[metric]:.4f}" for metric in metrics
                )
            )
            logger.info(
                f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, "
                + ", ".join(
                    f"{metric}: {val_metrics[metric]:.4f}" for metric in metrics
                )
            )
            self.system_usage()

            # Save the best model
            if (
                self.config.best_model_metric_greater
                and val_metrics[self.config.best_model_metric] > best_metric_value
            ) or (
                not self.config.best_model_metric_greater
                and val_metrics[self.config.best_model_metric] < best_metric_value
            ):
                best_metric_value = val_metrics[self.config.best_model_metric]
                torch.save(self.model.state_dict(), best_model_path)
                torch.save(
                    self.model,
                    self.config.run_dir
                    / self.config.best_model_name.replace(".pth", "_full.pth"),
                )
                logger.info(f"Best model at {epoch} saved at {best_model_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement in {self.config.best_model_metric}. "
                    f"Patience counter: {self.patience_counter}/{self.config.early_stopping_patience}"
                )
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered. No improvement for "
                        f"{self.config.early_stopping_patience} epochs."
                    )
                    break
            # Save the optimizer state
            torch.save(
                self.optimizer.state_dict(),
                self.config.run_dir / "optimizer_state.pth",
            )
            # save last model
            torch.save(
                self.model.state_dict(),
                self.config.run_dir / "last_model.pth",
            )
            # save last model full
            torch.save(
                self.model,
                self.config.run_dir / "last_model_full.pth",
            )
            # update metric history
            for metric in epoch_metrics:
                self.metric_history[metric].append(epoch_metrics[metric])
            for metric in val_metrics:
                self.metric_history[f"val_{metric}"].append(val_metrics[metric])
            # Log metrics
            self.log_metrics(
                epoch,
                epoch_loss,
                epoch_metrics,
                val_loss,
                val_metrics,
            )
            # if lr gets 0, stop training
            if self.optimizer.param_groups[0]["lr"] <= 0:
                logger.info(
                    "Learning rate has reached 0. Stopping training to prevent further issues."
                )
                break
        logger.info("Training completed.")
        if self.config.log_mlflow:
            mlflow.end_run()
            logger.info("MLflow run ended.")
        return best_model_path

    def log_metrics(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        """
        Log the training and validation metrics.
        """
        # Save metrics to file: epoch, train_loss, train_metrics, val_loss, val_metrics
        metrics_file = self.config.run_dir / self.config.metric_file
        # if it is a first epoch and the file exists, write header
        metrics = val_metrics.keys()
        if epoch == 0:
            with open(metrics_file, "w") as f:
                header = (
                    "epoch,train_loss,"
                    + ",".join(metrics)
                    + ",val_loss,"
                    + ",".join(f"val_{metric}" for metric in metrics)
                )
                f.write(header + "\n")
        with open(metrics_file, "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},"
                + ",".join(f"{train_metrics[metric]:.4f}" for metric in metrics)
                + f",{val_loss},"
                + ",".join(f"{val_metrics[metric]:.4f}" for metric in metrics)
                + "\n"
            )
        logger.info(f"Metrics logged to {metrics_file}")
        if self.config.log_mlflow:
            # if mlflow object is not None, log metrics to mlflow
            if not self.started_mlflow:
                mlflow.set_tracking_uri("http://localhost:5000")
                mlflow.set_experiment(self.config.expt_name)
                mlflow.start_run(
                    run_name=self.config.run_name,
                    nested=False,
                )
                logger.info("MLflow run started.")
            self.started_mlflow = True
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for metric in metrics:
                mlflow.log_metric(f"train_{metric}", train_metrics[metric], step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for metric in metrics:
                mlflow.log_metric(f"val_{metric}", val_metrics[metric], step=epoch)
            logger.info("Metrics logged to MLflow.")

    def system_usage(self):
        """
        Log the system usage statistics.
        """
        import psutil

        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        gpu = None
        if torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / (1024**2)
        logger.info(
            f"System Usage - CPU: {cpu}%, Memory: {memory.percent}%, GPU: {gpu} MB"
            if gpu is not None
            else "GPU: Not available"
        )
        return {
            "cpu": cpu,
            "memory": memory.percent,
            "gpu": gpu if gpu is not None else "Not available",
        }

    def plot_metrics(self):
        plot_data = {
            "epoch": list(range(1, len(self.metric_history["loss"]) + 1)),
            "train_loss": self.metric_history["loss"],
            "val_loss": self.metric_history["val_loss"],
        }
        for metric in self.metrics:
            plot_data[metric] = self.metric_history[metric]
            plot_data[f"val_{metric}"] = self.metric_history[f"val_{metric}"]

        sns.set(style="whitegrid")
        df = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 8))
        for metric in self.metrics:
            plt.plot(
                df["epoch"],
                df[metric],
                label=metric,
                marker="o",
                linestyle="-",
            )
            plt.plot(
                df["epoch"],
                df[f"val_{metric}"],
                label=f"val_{metric}",
                marker="x",
                linestyle="--",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training and Validation Metrics")
        plt.legend()
        plt.grid()
        plt.savefig(self.config.run_dir / "metrics_plot.png")
        plt.close()

        logger.info(f"Metrics plot saved at {self.config.run_dir / 'metrics_plot.png'}")

        # plot losses
        plt.figure(figsize=(12, 6))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(self.config.run_dir / "loss_plot.png")
        plt.close()
        logger.info(f"Loss plot saved at {self.config.run_dir / 'loss_plot.png'}")
