import argparse

import pytorch_lightning as pl
import torch


def load_and_rename_checkpoint(input_checkpoint_path, output_checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(
        input_checkpoint_path, map_location=lambda storage, loc: storage
    )

    replacements = [
        ("segmentation_model", "model"),
        ("encoder_name", "backbone"),
        ("encoder_weights", "weights"),
        ("learning_rate", "lr"),
        ("learning_rate_schedule_patience", "patience"),
    ]

    for src, dst in replacements:
        if "hyper_parameters" in checkpoint and src in checkpoint["hyper_parameters"]:
            checkpoint["hyper_parameters"][dst] = checkpoint["hyper_parameters"].pop(
                src
            )
            print(f"Renamed {src} to {dst}")

    # Save the checkpoint again with the new name
    torch.save(checkpoint, output_checkpoint_path)
    print(f"Checkpoint saved to {output_checkpoint_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load a PyTorch Lightning checkpoint, rename 'segmentation_model' to 'model', and save it again."
    )
    parser.add_argument(
        "input_checkpoint", type=str, help="Path to the input checkpoint file."
    )
    parser.add_argument(
        "output_checkpoint", type=str, help="Path to the output checkpoint file."
    )

    # Parse arguments
    args = parser.parse_args()

    # Load and rename the checkpoint
    load_and_rename_checkpoint(args.input_checkpoint, args.output_checkpoint)
