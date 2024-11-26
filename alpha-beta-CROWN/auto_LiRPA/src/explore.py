from verified_grid import VerifiedGrid
from verify import MNISTVerifier
from utils import load_image
class Explorer: 
    def __init__(self, verifier: MNISTVerifier, image_dims):
        self.verifier = verifier
        self.image_dims = image_dims
        self.verified_grid = VerifiedGrid(image_dims)
        self.target_k = None
        self.image = None
        
        self.unsafe_proved = False
        
        self.unsafe_corner = None
        
        print(f"Initialised Explorer")
        
    
    def recursive_explore(self, corner, k):
        print(f"Exploring {corner} with k={k}")
        if (self.unsafe_proved):
            print(f"Exiting, already proven unsafe")
            return
        if (self.verified_grid.is_verified(corner, k)):
            print(f"Exiting, corner {corner} already verified for k={k}")
            return
        verified = self.verifier.verify(corner, k, self.image)
        if (verified):
            self.verified_grid.add_verified(corner, k)
            print(f"Exiting, verified for {corner} with k={k}")
            return
        else:
            if k == self.target_k:
                self.unsafe_proved = True
                self.unsafe_corner = corner
                print(f"Exiting, unsafe region detected for {corner} with target k={k}")
                return
            smaller_k = max(k-1, self.target_k)
            x, y = corner
            for i in range(x, x + (k - smaller_k) + 1):
                for j in range(y, y + (k - smaller_k) + 1):
                    print(f"Recursively trying for smaller region {(i, j)} with smaller k={smaller_k}")
                    self.recursive_explore((i, j), smaller_k)

    def explore(self, image_path, k):
        self.unsafe_proved = False
        self.target_k = k
        if image_path:
            self.image = load_image(image_path)

        (x, y) = self.image_dims
        max_k = min(x, y)
        for i in range(0, x - max_k + 1):
            for j in range(0, y - max_k + 1):
                self.recursive_explore((i, j), max_k)
        return (not self.unsafe_proved)
            
    
        
        