import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base.modules import SCSEModule
from torch.nn.utils.rnn import pad_sequence

from rosaid.models.cnn import BlockCNN2D
from rosaid.models.cnn import FeatureCNN1D as CNN1D
from rosaid.models.packet_autoencoder import PacketAutoencoder
from rosaid.models.squeeze_excite import SqueezeExciteConv1D


def pad_to_max_len(inp: list[torch.Tensor]) -> torch.Tensor:
    """Pad to (batch, channels=1, T_max) for Conv1d."""

    max_len = max(t.shape[-1] for t in inp)
    padded_list = []

    for t in inp:
        curr_len = t.shape[-1]
        if curr_len < max_len:
            pad_width = max_len - curr_len
            # (1, Ti) → pad time → (1, T_max)
            padded_t = F.pad(t.unsqueeze(0), (0, pad_width), value=0)
            padded_list.append(padded_t)
        else:
            padded_list.append(t.unsqueeze(0)[:, :max_len])

    x = torch.cat(padded_list, dim=0)  # (batch, 1, T_max) - Conv1d ready!
    return x


class PacketModel(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (1, 30, 50),
        channels: list[int] = [8, 16, 32],
        feature_dim: int = 8,
        global_pool: str = "avg",
    ):
        super(PacketModel, self).__init__()
        """
        https://www.sciencedirect.com/topics/computer-science/average-packet-size
        A typical Ethernet packet size is around 1500 bytes, but this can vary significantly based on the type of network traffic.

        Working of a model:
        1. Input Layer: The model takes in raw packet data, which is represented as a 3D tensor with dimensions corresponding to the height, width, and channels of the packet data.
        2. Convolutional layers: The model consists of multiple convolutional layers that apply filters to the input data to extract relevant features. Each convolutional layer is followed by an activation function (ReLU) and a max-pooling layer to reduce the spatial dimensions of the data.
        3. Fully connected layers: After the convolutional layers, the model has fully connected layers that further process the extracted features. These layers are also followed by activation functions (ReLU) and dropout layers to prevent overfitting.
        4. Output Layer: The final layer of the model is a fully connected layer that produces the output, which is a single value in this case (output_size=1).
        """
        feature_extractor = []
        self.channels = channels
        self.global_pool_type = global_pool
        self.feature_dim = feature_dim

        for channel in channels:
            feature_extractor.append(
                nn.Conv2d(
                    in_channels=(
                        input_size[0]
                        if len(feature_extractor) == 0
                        else channels[channels.index(channel) - 1]
                    ),
                    out_channels=channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feature_extractor.append(nn.ReLU(inplace=True))
            # feature_extractor.append(nn.MaxPool2d(kernel_size=2))
            feature_extractor.append(SCSEModule(channel, reduction=channel // 2))

        self.features = nn.Sequential(*feature_extractor)
        if global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError("global_pool must be either 'max' or 'avg'")
        # self.squeeze_excite = SqueezeAndExcitation1d(
        #     channels=channels[-1], reduction=16
        # )
        self.output_layer = nn.Sequential(
            nn.Linear(channels[-1], feature_dim),
            # nn.Sigmoid(),
        )

    @property
    def summary(self) -> str:
        from torchsummary import summary

        return summary(self, (3, 23, 23), device="cpu")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: num_pkts, 3, 23, 23
        x = self.features(inp)
        x = self.global_pool(x)
        #
        # x = x.view(x.size(0), -1, 1)
        # x = self.squeeze_excite(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x


class DynamicPacketModel(nn.Module):
    def __init__(
        self,
        channels: list[int] = [
            64,
            32,
            64,
        ],
        feature_dim: int = 128,
        global_pool: str = "max",
    ):
        super(DynamicPacketModel, self).__init__()
        """
        A model that can take Dynamic input size by using adaptive pooling.
        """
        feature_extractor = []
        self.channels = channels
        self.global_pool_type = global_pool
        self.feature_dim = feature_dim
        # input to the model would be 1 channel, N packet length
        # conv1d layers
        in_channels = 1

        for channel in channels:
            feature_extractor.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feature_extractor.append(nn.ReLU(inplace=True))
            feature_extractor.append(SqueezeExciteConv1D(channel))
            in_channels = channel
        self.features = nn.Sequential(*feature_extractor)
        if global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        elif global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("global_pool must be either 'max' or 'avg'")
        self.output_layer = nn.Sequential(
            nn.Linear(channels[-1], feature_dim),
            # nn.Sigmoid(),
        )

    @property
    def summary(self) -> str:
        from torchsummary import summary

        return summary(self, (1, 1, 50), device="cpu")

    # def forward(self, inp: list[torch.Tensor]) -> torch.Tensor:
    #     # inp: num_pkts, 1, Dynamic pkt_length
    #     features = []
    #     # instead of looping, can we do multi processing here?

    #     for x in inp:
    #         x = self.features(x)
    #         x = self.global_pool(x)
    #         x = x.view(x.size(0), -1)
    #         x = self.output_layer(x)
    #         features.append(x)

    #     return torch.stack(features)

    def forward(self, inp) -> torch.Tensor:
        """
        Args:
            inp: Either:
                - List of tensors with shape (1, 1, N): [(1, 1, N1), (1, 1, N2), ...]
                - Batched tensor: (batch, 1, length)

        Returns:
            torch.Tensor of shape (num_packets, feature_dim)
        """
        # Handle variable-length list input
        if isinstance(inp, list):
            x = self._handle_variable_length(inp)
        else:
            x = inp

        # Feature extraction (now processes entire batch in parallel)
        x = self.features(x)  # (batch, channels[-1], L')
        x = self.global_pool(x)  # (batch, channels[-1], 1)
        x = x.squeeze(-1)  # (batch, channels[-1])
        x = self.output_layer(x)  # (batch, feature_dim)

        return x

    def _handle_variable_length(self, inp: list[torch.Tensor]) -> torch.Tensor:
        """
        Convert list of variable-length packets to padded batch.
        Handles input shape (1, 1, N).

        Args:
            inp: List of tensors with shape (1, 1, N) where N is Dynamic

        Returns:
            Padded batch tensor of shape (num_packets, 1, max_length)
        """
        # Step 1: Extract the sequence dimension (squeeze batch and channel)
        # Input: (1, 1, N) → Output: (N,)
        normalized = []
        for x in inp:
            if x.dim() == 3:  # Shape: (1, 1, length)
                x = x.squeeze(0).squeeze(0)  # → (length,)
            elif x.dim() == 2:  # Fallback: (1, length)
                x = x.squeeze(0)  # → (length,)
            normalized.append(x)

        # Step 2: Pad all packets to max length
        # Shorter packets get zero-padded at the end
        padded = pad_sequence(
            normalized, batch_first=True, padding_value=-0.1
        )  # (num_packets, max_length)

        # Step 3: Add channel dimension for Conv1d
        padded = padded.unsqueeze(1)  # (num_packets, 1, max_length)

        return padded


class PacketFlowClassifier(nn.Module):
    def __init__(
        self,
        packet_feature_extractor: nn.Module | PacketAutoencoder,
        in_channels: int = 3,
        output_size: int = 9,
        hidden_channels: int = [32, 64, 128, 64, 32],
    ):
        super(PacketFlowClassifier, self).__init__()
        self.pfe = packet_feature_extractor
        self.session_output = CNN1D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            output_size=output_size,
            dropout_rate=0.3,
            use_batchnorm=True,
            kernel_size=3,
        )
        self.softmax = nn.Softmax(dim=1)

    @property
    def summary(self) -> str:
        from torchsummary import summary

        return summary(self, (1, 30, 50), device="cpu")

    def forward(self, packet_features: torch.Tensor) -> torch.Tensor:
        packet_features = packet_features.view(
            1, -1, len(packet_features)
        )  # 1, feat_dim, num_pkts
        logits = self.session_output(
            packet_features
        )  # 1, feat_dim bcz CNN1D adds channel dim

        return logits, self.softmax(logits)


class SessionFeatureFusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list[int] = [8, 16, 32, 128, 256],
        num_classes: int = 2,
        clf_hidden_channels: list[int] = [8, 16, 32, 64],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.blocks = nn.ModuleDict()
        in_ch = in_channels

        for idx, ch in enumerate(hidden_channels):
            # conv with same padding
            layer = [
                nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ]
            if idx % 2 == 1:
                layer.append(SqueezeExciteConv1D(ch))
            block = nn.Sequential(*layer)
            self.blocks[f"block_{idx}"] = block
            in_ch = ch

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fusion_weights_x = nn.Parameter(torch.tensor([0.6, 0.4]))  # Packet
        self.fusion_weights_xt = nn.Parameter(torch.tensor([0.5, 0.5]))  # Byte
        self.fusion_gate = nn.Softmax(dim=0)

        self.classifier = BlockCNN2D(
            input_size=(1, ch, ch),
            output_size=num_classes,
            global_pool="avg",
            hidden_channels=clf_hidden_channels,
        )

    @property
    def summary(self) -> str:
        clf_params = sum(p.numel() for p in self.classifier.parameters())
        fusion_params = 4  # 2 weights each for x and x_t
        block_params = sum(
            p.numel() for block in self.blocks.values() for p in block.parameters()
        )
        total_params = clf_params + fusion_params + block_params
        return f"SessionFeatureFusionModel Summary:\nTotal Parameters: {total_params}\nClassifier Parameters: {clf_params}\nBlock Parameters: {block_params}\nFusion Parameters: {fusion_params}"

    def get_features(self, inp: list[torch.Tensor]) -> tuple:
        """
        inp: session with list[pkt1, pkt2, ...]
        Returns: (gram_matrix, logits)
        """
        # Packet-wise (horizontal)
        x = pad_to_max_len(inp)  # (N_packets, 1, T_max)
        x_t = x.transpose(2, 0)  # (T_max, 1, N_packets)

        # Pass through conv blocks
        pkts_feat = x.clone()
        for idx, block in self.blocks.items():
            pkts_feat = block(pkts_feat)  # (N_packets, 128, T_max)

        sess_feat = x_t.clone()
        for idx, block in self.blocks.items():
            sess_feat = block(sess_feat)  # (T_max, 128, N_packets)

        # print(f"x shape: {x.shape}, x_t shape: {x_t.shape}")

        # Pool across packet dimension (axis=0)
        x_avg = pkts_feat.mean(axis=0)
        x_max = pkts_feat.max(axis=0).values
        x_t_avg = sess_feat.mean(axis=2)
        x_t_max = sess_feat.max(axis=2).values

        # Independent fusion
        w_x = self.fusion_gate(self.fusion_weights_x)
        w_xt = self.fusion_gate(self.fusion_weights_xt)

        x_fused = w_x[0] * x_avg + w_x[1] * x_max  # (128, T_max)
        x_t_fused = w_xt[0] * x_t_avg + w_xt[1] * x_t_max  # (T_max, 128)

        gram_matrix = torch.matmul(x_fused, x_t_fused)

        # print(f"gram_matrix shape: {gram_matrix.shape}")

        return gram_matrix

    def classify(self, features: torch.Tensor) -> tuple:
        """
        Input: features: (B, 1, H, W)
        """
        logits, probs = self.classifier(features)
        return logits, probs

    def forward(self, batch: list[list[torch.Tensor]]) -> tuple:
        """
        inp: batch of sessions: list[session1, session2, ...]
            where session: list[pkt1, pkt2, ...]
        """
        features = [self.get_features(session) for session in batch]
        remove_last = False
        if len(features) == 1 and self.training:
            features = [features[0], features[0]]
            remove_last = True
        # make batch of gram matrices
        features = torch.stack(features)  # (batch_size, H, W)
        logits, probs = self.classify(features.unsqueeze(1))  # add channel dim
        if remove_last:
            features = features[:1]
            logits = logits[:1]
            probs = probs[:1]
        return features, (logits, probs)


if __name__ == "__main__":
    # packet_model = PacketModel()
    # print(packet_model.summary)
    # session_model = SessionFeatureExtractor()
    # input_tensor = torch.randn(32, 3, 23, 23)
    # output = session_model(input_tensor)
    # # print(output.shape)
    # print(session_model.summary)

    # input_packets = [
    #     torch.randn(1, 1, torch.randint(20, 100, (1,)).item()) for _ in range(10)
    # ]
    # packet_model = DynamicPacketModel()
    # output = packet_model(input_packets)
    # print("Output shape:", output.shape)

    sessions = [
        [
            torch.randn(1, 254),
            torch.randn(1, 25),
            torch.randn(1, 250),
            torch.randn(1, 250),
        ]
        for i in range(2)
    ]
    model = SessionFeatureFusionModel()
    gram_matrix, (logits, probs) = model(sessions)

    pass
