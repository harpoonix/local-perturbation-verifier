from torchvision.io import read_image
from torchvision.transforms import Compose, Grayscale
import torch

def load_image(image_path: str) -> torch.Tensor:
    try:
        image = read_image(image_path)
    except:
        print("Error reading image")
        exit(1);
    print("\nSuccessfully read image")
    transform = Compose([
        Grayscale(num_output_channels=1),
    ])
    try:
        image = transform(image).unsqueeze(0)
    except:
        print("Error transforming image to Grayscale")
        exit(1);
    image = image.to(torch.float32) / 255.0
    if torch.cuda.is_available():
        image = image.cuda()
    return image