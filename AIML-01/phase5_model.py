"""
PHASE 5: CNN MODEL ARCHITECTURE

Medical AI Analysis Platform
Task: Pediatric chest X-ray classification
Classes:
    0 = NORMAL
    1 = PNEUMONIA

Input:
    [batch_size, 1, 224, 224]

Output:
    [batch_size, 2] raw classification logits
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

class ModelConfig:
    IMAGE_CHANNELS = 1
    IMAGE_SIZE = 224
    NUM_CLASSES = 2

    CLASS_NAMES = [
        "NORMAL",
        "PNEUMONIA",
    ]


# ============================================================
# 2. CONVOLUTIONAL BLOCK
# ============================================================

class ConvBlock(nn.Module):
    """
    A reusable CNN block.

    Structure:
        Convolution
        Batch Normalization
        ReLU
        Convolution
        Batch Normalization
        ReLU
        Max Pooling
        Dropout
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.block = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second convolution
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Reduce image dimensions
            nn.MaxPool2d(kernel_size=2),

            # Reduce overfitting
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 3. COMPLETE CNN MODEL
# ============================================================

class ChestXRayCNN(nn.Module):
    """
    CNN for binary chest X-ray classification.
    """

    def __init__(self):
        super().__init__()

        # Feature extraction block 1
        self.block1 = ConvBlock(
            in_channels=1,
            out_channels=32,
            dropout_rate=0.10,
        )

        # Feature extraction block 2
        self.block2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            dropout_rate=0.15,
        )

        # Feature extraction block 3
        self.block3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            dropout_rate=0.20,
        )

        # Feature extraction block 4
        self.block4 = ConvBlock(
            in_channels=128,
            out_channels=256,
            dropout_rate=0.25,
        )

        # Convert each feature channel into one value
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.40),

            nn.Linear(128, 2),
        )

        # Initialize convolution and linear weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization.

        This initialization is suitable for networks
        using ReLU activation functions.
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):

        # Block 1
        x = self.block1(x)

        # Block 2
        x = self.block2(x)

        # Block 3
        x = self.block3(x)

        # Block 4
        x = self.block4(x)

        # Global average pooling
        x = self.global_pool(x)

        # Classification
        x = self.classifier(x)

        return x


# ============================================================
# 4. PARAMETER COUNT
# ============================================================

def count_parameters(model):
    """
    Count trainable and non-trainable parameters.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    non_trainable = total - trainable

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }


# ============================================================
# 5. MODEL SUMMARY
# ============================================================

def model_summary(model, device):
    """
    Print model configuration and parameter information.
    """

    parameters = count_parameters(model)

    print("\n========== PHASE 5 MODEL SUMMARY ==========")

    print(f"Model: {model.__class__.__name__}")

    print(
        f"Input shape: "
        f"[batch, {ModelConfig.IMAGE_CHANNELS}, "
        f"{ModelConfig.IMAGE_SIZE}, {ModelConfig.IMAGE_SIZE}]"
    )

    print(f"Output classes: {ModelConfig.NUM_CLASSES}")

    print(f"Class names: {ModelConfig.CLASS_NAMES}")

    print(f"Device: {device}")

    print(
        f"Total parameters: "
        f"{parameters['total_parameters']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{parameters['trainable_parameters']:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{parameters['non_trainable_parameters']:,}"
    )

    print("\nArchitecture:")
    print(model)

    return parameters


# ============================================================
# 6. SHAPE VERIFICATION
# ============================================================

def verify_model(model, device):
    """
    Verify:
        - Input shape
        - Output shape
        - Finite outputs
        - Backpropagation
        - Gradient flow
    """

    print("\n========== MODEL VERIFICATION ==========")

    model = model.to(device)

    batch_size = 4

    # Simulate four preprocessed X-ray images
    dummy_images = torch.randn(
        batch_size,
        ModelConfig.IMAGE_CHANNELS,
        ModelConfig.IMAGE_SIZE,
        ModelConfig.IMAGE_SIZE,
        device=device,
    )

    # Test inference
    model.eval()

    with torch.inference_mode():

        output = model(dummy_images)

    print("Input shape:", tuple(dummy_images.shape))

    print("Output shape:", tuple(output.shape))

    print("Output dtype:", output.dtype)

    # Verify expected output
    expected_shape = (
        batch_size,
        ModelConfig.NUM_CLASSES,
    )

    assert tuple(output.shape) == expected_shape, (
        f"Wrong output shape: {tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        "Model output contains NaN or infinity."
    )

    print("Output shape verification: PASSED")

    print("Finite output verification: PASSED")

    # Test training and gradient flow
    model.train()

    model.zero_grad(set_to_none=True)

    dummy_labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    train_output = model(dummy_images)

    loss = criterion(
        train_output,
        dummy_labels,
    )

    loss.backward()

    # Check whether gradients were calculated
    gradient_count = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
    )

    assert gradient_count > 0, (
        "No valid gradients were calculated."
    )

    print("Dummy training loss:", round(loss.item(), 6))

    print("Parameters with valid gradients:", gradient_count)

    print("Backpropagation verification: PASSED")

    model.zero_grad(set_to_none=True)

    if device.type == "cuda":

        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated(device) / (1024 ** 2),
                2,
            ),
            "MB",
        )

    print("\nMODEL VERIFICATION PASSED.")

    return True


# ============================================================
# 7. SAVE MODEL CONFIGURATION
# ============================================================

def save_model_config(model, output_path):
    """
    Save architecture metadata.

    This is configuration information, not trained weights.
    """

    parameters = count_parameters(model)

    report = {
        "phase": 5,
        "model_name": "ChestXRayCNN",
        "task": "Binary chest X-ray classification",
        "input_shape": [
            ModelConfig.IMAGE_CHANNELS,
            ModelConfig.IMAGE_SIZE,
            ModelConfig.IMAGE_SIZE,
        ],
        "output_classes": ModelConfig.CLASS_NAMES,
        "total_parameters": parameters["total_parameters"],
        "trainable_parameters": parameters["trainable_parameters"],
        "architecture": {
            "block1": "1 -> 32",
            "block2": "32 -> 64",
            "block3": "64 -> 128",
            "block4": "128 -> 256",
            "pooling": "Adaptive Average Pooling",
            "classifier": "256 -> 128 -> 2",
        },
        "status": "architecture_verified",
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\nModel configuration saved:", output_path)


# ============================================================
# 8. DEVICE SELECTION
# ============================================================

def get_device(device_option):

    if device_option == "cpu":
        return torch.device("cpu")

    if device_option == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access it."
            )

        return torch.device("cuda")

    # Automatic selection
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 5: Chest X-ray CNN Model"
    )

    parser.add_argument(
        "command",
        choices=[
            "summary",
            "test",
            "all",
        ],
        help="Operation to execute.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args()

    device = get_device(args.device)

    if device.type == "cuda":

        print("GPU:", torch.cuda.get_device_name(device))

    else:

        print("Using CPU")

    model = ChestXRayCNN()

    model = model.to(device)

    if args.command in ["summary", "all"]:

        model_summary(
            model,
            device,
        )

    if args.command in ["test", "all"]:

        verify_model(
            model,
            device,
        )

    if args.command == "all":

        save_model_config(
            model,
            "AIML-01/data/processed/phase5/model_config.json",
        )


if __name__ == "__main__":
    main()
