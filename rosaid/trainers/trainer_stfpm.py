import sys
from collections import deque

import torch
from tqdm import tqdm

from rosaid.data.dataset import DFDataSet
from rosaid.data.flow_dataset import CLFDataSet
from rosaid.loss.stfpm_loss import STFPMLoss
from rosaid.models.stfpm import STFPModel
from rosaid.trainers.trainer import NNTrainer, NNTrainerConfig


class STFPMTrainerConfig(NNTrainerConfig):
    pass


class STFPMTrainer(NNTrainer):
    def __init__(
        self,
        config: STFPMTrainerConfig,
        model: STFPModel,
        train_dataset: CLFDataSet | DFDataSet,
        val_dataset: CLFDataSet | DFDataSet,
        criterion=STFPMLoss(),
    ):
        super().__init__(config, model, train_dataset, val_dataset, criterion)

        self.model.to(self.device)
        self.train_clean_inp_lbl_buffer = deque(maxlen=100)
        self.val_clean_inp_lbl_buffer = deque(maxlen=100)

    # def at_batch_end(self):
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def forward_step(self, batch):
        inputs, label_tensor = batch
        # batch_logits = []
        batch_probs = []

        session_tensor = inputs
        curr_clean_samples = []
        curr_attack_samples = []
        # put in buffer
        if self.is_train:
            for i in range(len(label_tensor)):
                if label_tensor[i].argmax() == 0:
                    self.train_clean_inp_lbl_buffer.append(
                        (session_tensor[i].cpu(), label_tensor[i].cpu())
                    )
                    curr_clean_samples.append(
                        (session_tensor[i].cpu(), label_tensor[i].cpu())
                    )
                else:
                    curr_attack_samples.append(
                        (session_tensor[i].cpu(), label_tensor[i].cpu())
                    )
        else:
            # during validation, use both
            for i in range(len(label_tensor)):
                if label_tensor[i].argmax() == 0:
                    self.val_clean_inp_lbl_buffer.append(
                        (session_tensor[i].cpu(), label_tensor[i].cpu())
                    )
                    curr_clean_samples.append(
                        (session_tensor[i].cpu(), label_tensor[i].cpu())
                    )
                else:
                    curr_attack_samples.append(
                        (session_tensor[i].cpu(), label_tensor[i].cpu())
                    )
        # now pass through model
        if self.is_train:
            # make new batch with len(inputs)
            # sample from buffer to match current batch size
            num_needed = len(inputs) - len(curr_clean_samples)
            if num_needed > 0:
                if num_needed > len(self.train_clean_inp_lbl_buffer):
                    replace = True
                else:
                    replace = False
                clean_idxs = self.random_state.choice(
                    torch.arange(len(self.train_clean_inp_lbl_buffer)),
                    size=num_needed,
                    replace=replace,
                )
                sampled_clean = [
                    self.train_clean_inp_lbl_buffer[idx] for idx in clean_idxs
                ]

            else:
                sampled_clean = []
            final_batch = curr_clean_samples + sampled_clean
            clean_session_tensor = torch.stack(
                [item[0] for item in final_batch], dim=0
            ).to(self.device)

            clean_teacher_features, clean_student_features = self.model(
                clean_session_tensor
            )
            loss = self.criterion(clean_teacher_features, clean_student_features)

            metrics = dict()
            metrics["clean_loss"] = loss.item()
            metrics["loss"] = loss.item()
            metrics["attack_loss"] = 0.0
        else:
            # for validation use make distinct clean and attack batch and find distance loss
            clean_samples = []
            attack_samples = []
            for item in curr_clean_samples:
                clean_samples.append(item)
            for item in curr_attack_samples:
                attack_samples.append(item)
            # if either is empty, sample from buffer
            if len(clean_samples) == 0:
                num_needed = len(attack_samples)
                replace = len(self.val_clean_inp_lbl_buffer) < num_needed
                clean_idxs = self.random_state.choice(
                    torch.arange(len(self.val_clean_inp_lbl_buffer)),
                    size=num_needed,
                    replace=replace,
                )
                sampled_clean = [
                    self.val_clean_inp_lbl_buffer[idx] for idx in clean_idxs
                ]
                clean_samples.extend(sampled_clean)
            if len(attack_samples) == 0:
                num_needed = len(clean_samples)
                attack_buffer_size = len(self.train_clean_inp_lbl_buffer)
                replace = attack_buffer_size < num_needed
                attack_idxs = self.random_state.choice(
                    torch.arange(attack_buffer_size),
                    size=num_needed,
                    replace=replace,
                )
                sampled_attack = [
                    self.train_clean_inp_lbl_buffer[idx] for idx in attack_idxs
                ]
                attack_samples.extend(sampled_attack)

            clean_session_tensor = torch.stack(
                [item[0] for item in clean_samples], dim=0
            ).to(self.device)
            attack_session_tensor = torch.stack(
                [item[0] for item in attack_samples], dim=0
            ).to(self.device)

            teacher_feature, student_feature = self.model(clean_session_tensor)
            # average feature losses
            clean_loss = self.criterion(teacher_feature, student_feature)

            attack_teacher_features, attack_student_features = self.model(
                attack_session_tensor
            )
            # average feature losses
            attack_loss = self.criterion(
                attack_teacher_features, attack_student_features
            )

            metrics = dict()
            metrics["clean_loss"] = clean_loss.item()
            metrics["attack_loss"] = attack_loss.item()
            # bcz we minimize clean loss
            loss = clean_loss
            metrics["loss"] = loss.item()

        return batch_probs, loss, metrics

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
            if metric in epoch_metrics:
                epoch_metrics[metric] /= len(dataloader)
            else:
                epoch_metrics[metric] = metrics[metric] / len(dataloader)
        epoch_metrics["loss"] = epoch_loss
        return outputs, epoch_loss, epoch_metrics
