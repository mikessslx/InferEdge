"""This module downloads and converts selected MobileNetV3 model variants to Torchscript format."""
import os
import torch
from torch import jit
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
import argparse

def main():
    # Parse input arguments representing which variants to generate
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobilenetv3_small", action="store_true", help="Generate MobileNetV3-Small")
    parser.add_argument("--mobilenetv3_large", action="store_true", help="Generate MobileNetV3-Large")
    args = parser.parse_args()

    mobilenet_variants = []
    mobilenet_variants_suffixes = []

    if args.mobilenetv3_small:
        mobilenet_variants.append(mobilenet_v3_small)
        mobilenet_variants_suffixes.append("small")
    if args.mobilenetv3_large:
        mobilenet_variants.append(mobilenet_v3_large)
        mobilenet_variants_suffixes.append("large")

    generate_mobilenet_models(mobilenet_variants, mobilenet_variants_suffixes)
    
def generate_mobilenet_models(mobilenet_variants, mobilenet_variants_suffixes):
    # Create a dummy input
    fake_input = torch.rand(1, 3, 224, 224)

    for mobilenet_variant, mobilenet_variant_suffix in zip(mobilenet_variants, mobilenet_variants_suffixes):
        # Load the pretrained model
        model = mobilenet_variant(pretrained=True)
        model.eval()
        
        # Convert the model into a TorchScript module using tracing,
        # passing in the dummy input to trace
        traced_model = jit.trace(model, fake_input)

        # Freeze the model to optimize it
        frozen_model = torch.jit.freeze(traced_model)
        
        # Save the frozen model
        filename = f"mobilenetv3_{mobilenet_variant_suffix}.pt"
        frozen_model.save(filename)

if __name__ == "__main__":
    main()