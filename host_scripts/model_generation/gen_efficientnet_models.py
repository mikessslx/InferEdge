"""This module downloads and converts selected EfficientNet model variants to Torchscript format."""
import torch
from torch import jit
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
import argparse

def main():
    # Parse input arguments representing which variants to generate
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", action="store_true", help="Generate EfficientNet-B0")
    parser.add_argument("--b1", action="store_true", help="Generate EfficientNet-B1")
    parser.add_argument("--b2", action="store_true", help="Generate EfficientNet-B2")
    parser.add_argument("--b3", action="store_true", help="Generate EfficientNet-B3")
    parser.add_argument("--b4", action="store_true", help="Generate EfficientNet-B4")
    parser.add_argument("--b5", action="store_true", help="Generate EfficientNet-B5")
    parser.add_argument("--b6", action="store_true", help="Generate EfficientNet-B6")
    parser.add_argument("--b7", action="store_true", help="Generate EfficientNet-B7")
    args = parser.parse_args()

    efficientnet_variants = []
    efficientnet_numbers = []

    # TODO: add names as well
    if args.b0:
        efficientnet_variants.append(efficientnet_b0)
        efficientnet_numbers.append(0)
    if args.b1:
        efficientnet_variants.append(efficientnet_b1)
        efficientnet_numbers.append(1)
    if args.b2:
        efficientnet_variants.append(efficientnet_b2)
        efficientnet_numbers.append(2)
    if args.b3:
        efficientnet_variants.append(efficientnet_b3)
        efficientnet_numbers.append(3)
    if args.b4:    
        efficientnet_variants.append(efficientnet_b4)
        efficientnet_numbers.append(4)
    if args.b5:
        efficientnet_variants.append(efficientnet_b5)
        efficientnet_numbers.append(5)
    if args.b6:
        efficientnet_variants.append(efficientnet_b6)
        efficientnet_numbers.append(6)
    if args.b7:   
        efficientnet_variants.append(efficientnet_b7)
        efficientnet_numbers.append(7)

    generate_efficientnet_models(efficientnet_variants, efficientnet_numbers)
    
def generate_efficientnet_models(efficientnet_variants, efficientnet_numbers):
    # Create a dummy input 
    fake_input = torch.rand(1, 3, 224, 224)

    for efficientnet_variant, efficientnet_number in zip(efficientnet_variants, efficientnet_numbers):
        # Load the pretrained model
        model = efficientnet_variant(pretrained=True)
        model.eval()

        # Convert the model into a TorchScript module using tracing,
        # passing in the dummy input to trace
        traced_model = torch.jit.trace(model, fake_input)

        # Freeze the model to optimize it
        frozen_model = torch.jit.freeze(traced_model)

        # Save the frozen model
        filename = f"efficientnet_b{efficientnet_number}.pt"
        frozen_model.save(filename)

if __name__ == "__main__":
    main()