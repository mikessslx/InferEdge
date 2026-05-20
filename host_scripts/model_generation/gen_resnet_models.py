"""This module downloads and converts selected ResNet model variants to Torchscript format."""
import torch
from torch import jit
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import argparse

def main():
    # Parse input arguments representing which variants to generate
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet18", action="store_true", help="Generate ResNet-18")
    parser.add_argument("--resnet34", action="store_true", help="Generate ResNet-34")
    parser.add_argument("--resnet50", action="store_true", help="Generate ResNet-50")
    parser.add_argument("--resnet101", action="store_true", help="Generate ResNet-101")
    parser.add_argument("--resnet152", action="store_true", help="Generate ResNet-152")
    args = parser.parse_args()

    resnet_variants = []
    resnet_numbers = []

    if args.resnet18:
        resnet_variants.append(resnet18)
        resnet_numbers.append(18)
    if args.resnet34:
        resnet_variants.append(resnet34)
        resnet_numbers.append(34)
    if args.resnet50:
        resnet_variants.append(resnet50)
        resnet_numbers.append(50)
    if args.resnet101:
        resnet_variants.append(resnet101)
        resnet_numbers.append(101)
    if args.resnet152:
        resnet_variants.append(resnet152)
        resnet_numbers.append(152)

    generate_resnet_models(resnet_variants, resnet_numbers)

def generate_resnet_models(resnet_variants, resnet_numbers):
    # Create a dummy input
    fake_input = torch.rand(1, 3, 224, 224)

    for resnet_variant, resnet_number in zip(resnet_variants, resnet_numbers):
        # Load the pretrained model
        model = resnet_variant(pretrained=True)
        model.eval()

        # Convert the model into a TorchScript module using tracing,
        # passing in the dummy input to trace
        traced_model = jit.trace(model, fake_input)

        # Freeze the model to optimize it
        frozen_model = torch.jit.freeze(traced_model)

        # Save the frozen model
        filename = f"resnet{resnet_number}.pt"
        frozen_model.save(filename)

if __name__ == "__main__":
    main()