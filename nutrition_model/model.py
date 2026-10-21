"""MobileNetV3 encoder and four-output nutrition regression model."""
import torch
from torch import nn
import timm

BACKBONE = "mobilenetv3_large_100.ra_in1k"
NUM_TARGETS = 4


def load_encoder(pretrained=True):
    # num_classes=0 removes the original image-classification output.
    # Do not fall back to random weights if the download fails.
    return timm.create_model(
        BACKBONE,
        pretrained=pretrained,
        num_classes=0,
    )


class NutritionRegressor(nn.Module):
    """Predict four normalized nutrition targets from an RGB image batch."""

    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()
        self.encoder = load_encoder(pretrained=pretrained)
        self.feature_dim = self.encoder.head_hidden_size
        self.regression_head = nn.Linear(self.feature_dim, NUM_TARGETS)
        self.encoder_is_frozen = freeze_encoder

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm layers must keep their pretrained running statistics.
        if self.encoder_is_frozen:
            self.encoder.eval()
        return self

    def forward(self, images):
        if self.encoder_is_frozen:
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)
        return self.regression_head(features)


if __name__ == "__main__":
    torch.set_num_threads(2)
    model = NutritionRegressor(pretrained=True, freeze_encoder=True)
    model.eval()

    with torch.inference_mode():
        predictions = model(torch.zeros(2, 3, 224, 224))

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert predictions.shape == (2, NUM_TARGETS)
    assert torch.isfinite(predictions).all()
    assert trainable == model.feature_dim * NUM_TARGETS + NUM_TARGETS

    print(f"Pretrained encoder loaded: {BACKBONE}")
    print(f"Encoder features per image: {model.feature_dim}")
    print(f"Batch predictions: {tuple(predictions.shape)}")
    print(f"Trainable regression-head parameters: {trainable}")
