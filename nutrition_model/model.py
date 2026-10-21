"""Step 1: load a pretrained image encoder; no nutrition training yet."""
import torch
import timm

BACKBONE = 'mobilenetv3_large_100.ra_in1k'


def load_encoder():
    # num_classes=0 removes the original image-classification output.
    # Do not fall back to random weights if the download fails.
    return timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval()


if __name__ == '__main__':
    torch.set_num_threads(2)
    encoder = load_encoder()
    with torch.inference_mode():
        features = encoder(torch.zeros(1, 3, 224, 224))
    assert features.ndim == 2 and features.shape[0] == 1
    print(f'Pretrained encoder loaded: {BACKBONE}')
    print(f'One image -> feature vector: {tuple(features.shape)}')
    print('These features are NOT nutrition predictions.')
