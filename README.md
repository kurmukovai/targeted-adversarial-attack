# Targeted Adversarial Attack Tool

A Python tool for generating targeted adversarial examples for image classification models using Projected Gradient Descent (PGD) attacks with L infinity norm constraints.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/targeted-adversarial.git
cd targeted-adversarial

# Install the package
pip install -e .
```

## Usage

The tool provides a command-line interface for generating targeted adversarial examples:

```bash
# List all available ImageNet classes
targeted-adversarial list-classes

# Generate an adversarial example
targeted-adversarial run-attack path/to/image.jpg target_class_id --epsilon 0.1
```

### Example

```bash
# Generate an adversarial example to make a dog image be classified as a cat
targeted-adversarial run-attack assets/sample_images/dog.jpg n02123045
```

### Parameters

- `epsilon`: Maximum perturbation size (L∞ norm) [default: 0.1]
- `alpha`: Step size for each iteration [default: 0.03]
- `num_iter`: Number of PGD iterations [default: 40]
- `device`: Inference device (cuda or cpu) [default: cuda]
- `output`: Path to save the adversarial example [default: source_adversarial.jpg]


## Dataset

The tool includes sample images in `targeted_adversarial/assets/sample_images/` and uses the ImageNet class mapping stored in `targeted_adversarial/assets/LOC_synset_mapping.txt`.
