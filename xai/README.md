## Reasoning

Most of the avaliable explaination tools do not provide any meaningful ways to perform stability reaserch of the model explanations. 
We hope to bridge that gap by presenting model and method agnostic framework for quantifying the stability of model and quality of explanations.
## Instalation process

In order to use the library, installing required packages is necesseary.
It can be done via pip or any other preffered tooling:
```python
pip install -r requirements
```

## Library usage

In order to use the library, one must provide paths to the images in form of a csv file for in-built data loader. 
The example of the csv file is avaliable in xai_labels.csv file

The second component is configuration of the model and dataset main path, although the model can be easily replaced by specify inference function. 
The example of the config can be accessed in: xai/configs/xai_base.json

After the configuration and installation process is performed, the usage is as simple as running the command:
```bash
python inference_with_xai.py
```

After explanations are obtained, is is neccesseary to calculate statistics using explanation_analysis_notebook