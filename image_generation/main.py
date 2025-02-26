import os
from ecg_image_generator.gen_ecg_image_from_data import run_single_file
from pathlib import Path
configs = {
    "hw_test": {
        "hw_text": True
    },
    "winkles": {
        "wrinkles": True,
        "crease_angle": 45,
    }
}

filenames = [
    r"./img_to_modify/00040_hr-1.png",
    r"./img_to_modify/00057_hr-1.png"
]

def ensure_path_exists(file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def augment_image(filename, config, main_dir):
    output_image = os.path.join(main_dir, filename.split("/")[-1])
    output_image = str(Path(output_image).resolve())
    print(f"Output_image: {output_image}")
    ensure_path_exists(output_image)
    run_single_file(
        input_file=filename,
        output_file=output_image,
        **config
    )

def run_augmentation(main_dir):
    for filename in filenames:
        for config_name, config in configs.items():
            save_path = os.path.join(main_dir, config_name)
            print(f"Generating image for config: {config_name} and path: {save_path}")
            augment_image(filename, config, save_path)


if __name__ == "__main__":
    run_augmentation("./augmented_images")
