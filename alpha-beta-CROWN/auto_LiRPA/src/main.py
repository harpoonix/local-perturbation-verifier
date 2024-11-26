import argparse
from verify import MNISTVerifier
from explore import Explorer

if (__name__ == "__main__"):
    # model?
    # image?
    
    parser = argparse.ArgumentParser(
            prog = "LocalVerifier",
            description="Verifes robustness of neural networks against local perturbations"
        )
        
    parser.add_argument('-k', required=True, type=int, help="side length of subimage which can be perturbed")
    parser.add_argument('--epsilon', required=True, type=float, help="Norm bound on perturbation")
    def parse_p(value):
        if value == 'inf':
            return float('inf')
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value for -p: {value}")

    parser.add_argument('-p', required=True, type=parse_p, help="parameter p in Lp norm (can be 'inf' for infinity)")
    parser.add_argument('--model-checkpoint', required=False, type=str, help="Path to checkpoint of trained model")
    parser.add_argument('--image-path', required=False, type=str, help="Path of image on which robustness is being tested")
    
    args = parser.parse_args()
    
    verifier = MNISTVerifier(None, args.model_checkpoint, args.epsilon, args.p, 10)
    explorer = Explorer(verifier, (28, 28))
    
    # TODO: Add image to argument
    model_is_safe = explorer.explore(args.image_path, args.k)
    if (model_is_safe):
        print(f"Model is robust against perturbations for the given parameters")
    else:
        print(f"Model is not robust against perturbations to the patch of side-length {args.k} cornered at {explorer.unsafe_corner}")
    
    
