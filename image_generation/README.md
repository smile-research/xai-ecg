## Reasoning

The most well known [tool](https://github.com/alphanumericslab/ecg-image-kit) for generating augmented ecg scans is known for its limited interface. Although usefull, tool in its base form only works on signal data, generating augmentations in on-the-fly generated images. 

## Instalation process

In order to use the library, installing required packages is necesseary.
It can be done via pip or any other preffered tooling:
```python
pip install -r requirements
```

## Library usage

The exemplary usage of the tool is presented in main.py
It takes 2 main sources of configuration.
* List of files to augment
```python
filenames = [
    r"./img_to_modify/00040_hr-1.png",
    r"./img_to_modify/00057_hr-1.png"
]
```
* List of augmentation configs
```python
configs = {
    "hw_test": {
        "hw_text": True
    },
    "winkles": {
        "wrinkles": True,
        "crease_angle": 45,
    }
}
```
The arguments passed to the configs are the same as specified [here](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator).