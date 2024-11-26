import torch

class WrapperModel(torch.nn.Module):
    def __init__(self, model, image_dim, subimage_left_top_corner, subimage_dim):
        
        super().__init__()
        self.model = model
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.sub_image_corner_x, self.sub_image_corner_y = subimage_left_top_corner
        self.sub_image_width, self.sub_image_height = subimage_dim
        
        self.grad_bitmask = torch.zeros(image_dim)
        self.grad_bitmask[self.sub_image_corner_x: self.sub_image_corner_x + self.sub_image_width, self.sub_image_corner_y: self.sub_image_corner_y + self.sub_image_height] = 1

    def clip_grad(self, grad):
        print("Clipping gradient of input")
        return (grad * self.grad_bitmask.unsqueeze(0))
    
    def forward(self, image: torch.Tensor):
        # grad : b * w * h
        image.register_hook(self.clip_grad)
        return self.model(image)
        
        
