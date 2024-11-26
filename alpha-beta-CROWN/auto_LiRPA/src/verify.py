import torch
import torch.nn as nn
from auto_LiRPA.utils import Flatten
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

class Verifier:
    def __init__(self):
        print(f"Initialised Verifier")

    def verify(self, corner, k, image):
        raise NotImplemented

class MNISTVerifier(Verifier):
    def __init__(self, model, model_checkpoint_path, eps, norm, n_classes):
        super().__init__()
        if model is None:
            self.model = self._get_mnist_model()
        else:
            self.model = model
        if model_checkpoint_path == None:
            model_checkpoint_path = '../examples/vision/pretrained/mnist_a_adv.pth'
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        test_data = torchvision.datasets.MNIST(
            '../examples/vision/data', train=False, download=True,
            transform=torchvision.transforms.ToTensor())
        # For illustration we only use 1 image from dataset
        N = 1
        self.n_classes = n_classes
        self.sample_image = test_data.data[:N].view(N,1,28,28)
        # Convert to float
        self.sample_image = self.sample_image.to(torch.float32) / 255.0
        true_label = test_data.targets[:N]
        if torch.cuda.is_available():
            self.sample_image = self.sample_image.cuda()
            self.model = self.model.cuda()
        self.lirpa_model = BoundedModule(self.model, torch.empty_like(self.sample_image), device=self.sample_image.device)
        self.eps = eps
        self.norm = norm
        self.ptb = PerturbationLpNorm(norm = norm, eps = eps)
        self.sample_image = BoundedTensor(self.sample_image, self.ptb)

    def _get_mnist_model(self):
        model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32*7*7,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        return model
    
    def verify(self, corner, k, image=None):
        if image is None:
            image = self.sample_image
        else:
            image = BoundedTensor(image, self.ptb)
        bitmask = torch.zeros_like(image)
        bitmask[:, :, corner[0]:corner[0] + k, corner[1]:corner[1] + k] = 1
        bitmask = bitmask.flatten()
        
        self.ptb.bitmask = bitmask
        pred = self.lirpa_model(image)
        label = torch.argmax(pred, dim=1).cpu().detach().numpy()
        self.lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        C = torch.zeros(size=(1, self.n_classes - 1, self.n_classes), device=image.device)
        for i in range(self.n_classes):
            if i < label:
                C[0, i, label] = 1
                C[0, i, i] = -1
            elif i > label:
                C[0, i - 1, label] = 1
                C[0, i - 1, i] = -1
        lb, ub = self.lirpa_model.compute_bounds(x=(image,), method='CROWN-Optimized', C=C)
        print("********************************************************")
        all_verified = True
        for i in range(self.n_classes - 1):
            actual_idx = i if i < label else i + 1
            print(f'Lower bound for f{label.item()} - f{actual_idx}: {lb[0][i].item()}')
            all_verified = all_verified and (lb[0][i].item() >= 0)
        return all_verified
        
        
        