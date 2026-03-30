"""
Standalone augmentation tool that applies sequential augmentation steps
to arbitrary images, driven by a JSON config file.

Usage:
    uv run python apply_augmentations.py \
        --input_dir <dir> --output_dir <dir> --config <config.json>
"""

import argparse
import json
import os
import random
import shutil
import sys

from imgaug import augmenters as iaa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to the ecg-image-generator directory (sibling repo)
ECG_IMAGE_GENERATOR_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "augmentation_generation",
    "ecg-image-kit",
    "codes",
    "ecg-image-generator",
)
ECG_IMAGE_GENERATOR_DIR = os.path.abspath(ECG_IMAGE_GENERATOR_DIR)


def apply_wrinkles(image_path, params):
    """Apply wrinkles and/or creases overlay using get_creased().

    get_creased() writes in-place and requires CWD to be ecg-image-generator/
    for wrinkles-dataset access.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(ECG_IMAGE_GENERATOR_DIR)
        sys.path.insert(0, ECG_IMAGE_GENERATOR_DIR)
        from CreasesWrinkles.creases import get_creased

        output_directory = os.path.dirname(image_path)
        get_creased(
            input_file=image_path,
            output_directory=output_directory,
            ifWrinkles=params.get("ifWrinkles", True),
            ifCreases=params.get("ifCreases", True),
            crease_angle=params.get("crease_angle", 0),
            num_creases_vertically=params.get("num_creases_vertically", 3),
            num_creases_horizontally=params.get("num_creases_horizontally", 2),
        )
    finally:
        os.chdir(original_cwd)

    return image_path


def apply_augment(image_path, params):
    """Apply rotation, noise, crop, and color temperature augmentations.

    Simplified reimplementation of get_augment() without bbox/leads dependency.
    """
    image = Image.open(image_path)
    image = np.array(image)[:, :, :3]

    rotate = params.get("rotate", 25)
    noise = params.get("noise", 25)
    crop = params.get("crop", 0.01)
    temperature = params.get("temperature", 6500)

    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)

    seq = iaa.Sequential(
        [
            iaa.Affine(rotate=rot),
            iaa.AdditiveGaussianNoise(scale=(noise, noise)),
            iaa.Crop(percent=crop_sample),
            iaa.ChangeColorTemperature(temperature),
        ]
    )

    images_aug = seq(images=[image])
    plt.imsave(fname=image_path, arr=images_aug[0])

    return image_path


def apply_handwritten_text(image_path, params):
    """Add handwritten text overlay using get_handwritten().

    Requires CWD to be ecg-image-generator/ for model/data access.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(ECG_IMAGE_GENERATOR_DIR)
        sys.path.insert(0, ECG_IMAGE_GENERATOR_DIR)
        from HandwrittenText.generate import get_handwritten

        output_dir = os.path.dirname(image_path)
        get_handwritten(
            link=params.get("link", ""),
            num_words=params.get("num_words", 3),
            input_file=image_path,
            output_dir=output_dir,
            x_offset=params.get("x_offset", 0),
            y_offset=params.get("y_offset", 0),
            handwriting_size_factor=params.get("handwriting_size_factor", 0.2),
        )
    finally:
        os.chdir(original_cwd)

    return image_path


# Map config type names to handler functions
AUGMENTATION_HANDLERS = {
    "wrinkles": apply_wrinkles,
    "augment": apply_augment,
    "handwritten_text": apply_handwritten_text,
}


def process_image(image_path, image_dir, steps):
    """Copy image to output dir, apply each step sequentially, saving intermediates.

    Directory structure per image:
        <image_dir>/
            0_original.png
            1_wrinkles.png
            2_augment.png
            ...
            final.png
    """
    name_stem, name_ext = os.path.splitext(os.path.basename(image_path))

    # Save original copy as step 0
    original_path = os.path.join(image_dir, f"0_original{name_ext}")
    shutil.copy2(image_path, original_path)

    current_path = original_path
    for i, step in enumerate(steps, start=1):
        step_type = step["type"]
        handler = AUGMENTATION_HANDLERS.get(step_type)
        if handler is None:
            print(f"    WARNING: Unknown augmentation type '{step_type}', skipping.")
            continue

        # Copy current state to the next step file before augmenting in-place
        step_path = os.path.join(image_dir, f"{i}_{step_type}{name_ext}")
        shutil.copy2(current_path, step_path)

        params = {k: v for k, v in step.items() if k != "type"}
        print(f"    Step {i}/{len(steps)}: {step_type}")
        handler(step_path, params)
        current_path = step_path

    # Copy last step as final result
    final_path = os.path.join(image_dir, f"final{name_ext}")
    shutil.copy2(current_path, final_path)


def main():
    parser = argparse.ArgumentParser(
        description="Apply sequential augmentations to images from a JSON config."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory with input images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for augmented images."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="JSON config file with augmentation steps."
    )
    args = parser.parse_args()

    # Use matplotlib non-interactive backend
    matplotlib.use("Agg")

    # Load config
    with open(args.config, "r") as f:
        steps = json.load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy config into output dir for reproducibility
    config_copy = os.path.join(args.output_dir, "config.json")
    shutil.copy2(args.config, config_copy)

    # Find all image files in input directory
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_files = sorted(
        f
        for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in supported_extensions
    )

    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return

    # Print run summary
    step_names = [s["type"] for s in steps]
    print(f"Pipeline: {' -> '.join(step_names)}")
    print(f"Images:   {len(image_files)}")
    print(f"Output:   {args.output_dir}/")
    print()

    for idx, filename in enumerate(image_files, start=1):
        input_path = os.path.abspath(os.path.join(args.input_dir, filename))
        name_stem = os.path.splitext(filename)[0]

        # Each image gets its own subdirectory
        image_dir = os.path.abspath(os.path.join(args.output_dir, name_stem))
        os.makedirs(image_dir, exist_ok=True)

        print(f"  [{idx}/{len(image_files)}] {filename}")
        process_image(input_path, image_dir, steps)

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
