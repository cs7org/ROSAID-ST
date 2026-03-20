import torch

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

models = [
    'rosaid/results/image_classification/ROSIDS23_ORIGINAL/blockcnn2d_nosampling_binary_20260124/best_model_full.pth',
    'rosaid/results/image_classification/ROSIDS23_ORIGINAL/resnet18_nosampling_binary_20260124/best_model_full.pth',
    'rosaid/results/image_classification/ROSIDS23_ORIGINAL/mobilenet_v3_large_nosampling_binary_20260124/best_model_full.pth'
]
for model_path in models:
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    num_params = count_parameters(model)
    print(f"Number of trainable parameters in {model_path}: {num_params}")

