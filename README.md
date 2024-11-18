Link to github: https://github.com/harpoonix/local-perturbation-verifier

# Instructions to run the code

## Installation

```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
conda env create -f alpha-beta-CROWN/complete_verifier/environment.yaml --name alpha-beta-crown
# activate the environment
conda activate alpha-beta-crown
```

## Running the code

```bash
export PYTHONPATH=alpha-beta-CROWN/auto_LiRPA
cd alpha-beta-CROWN/auto_LiRPA/src
python main.py -k 14 --epsilon 0.35 -p 2
```
Optional arguments: 
- `--model-checkpoint`, specify the path to a trained model checkpoint. Can specify a custom `torch.nn.Module` model, and pass it as argument to instantiation of `MNISTVerifier` class in `main.py`  
- `--image-path`, specify the path to an image to verify. Default is taken from the MNIST test set.