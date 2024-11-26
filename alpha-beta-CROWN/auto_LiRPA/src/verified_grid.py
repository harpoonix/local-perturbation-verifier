import numpy as np

class VerifiedGrid:
    def __init__(self, image_dims):
        self.verified = np.zeros(shape=image_dims)
    
    def is_verified(self, corner, side_length):
        return self.verified[corner] >= side_length
    
    def add_verified(self, corner, side_length):
        x, y = corner
        for i in range(x, x + side_length):
            for j in range(y, y + side_length):
                self.verified[i, j] = max(self.verified[i, j], min(x + side_length - i, y + side_length - j))
        print(f"Verified grid:\n{self.verified}")
    