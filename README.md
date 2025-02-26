# README: Explainable AI for ECG Classification

## Motivation
Cardiovascular diseases (CVDs) are a leading global cause of mortality, necessitating early detection for effective treatment. Electrocardiography (ECG) is widely used for CVD screening, and AI models can enhance diagnostic accuracy. However, trust in AI-driven diagnostics is often hindered by a lack of transparency. Explainable Artificial Intelligence (XAI) techniques can help medical professionals understand and verify AI model decisions. This repository provides tools for ECG image generation, deep learning-based classification, and explanation robustness analysis using LIME.

## Structure of the Repository
The repository contains two main subfolders:
- `xai/` - Explanation code using LIME and PyTorch.
- `image_generation/` - Tooling for on-the-fly ECG image modifications.

### XAI
This module implements explainability techniques using LIME to interpret deep learning models applied to ECG scans. The robustness of explanations is analyzed against real-world perturbations such as:
- Handwriting over ECG scans
- Wrinkles in paper printouts
- Discoloration and scanning artifacts

### Image Generation
This module facilitates the transformation of digital ECG signals into images with different visual modifications to simulate real-world distortions. This ensures the AI models generalize across diverse ECG printouts encountered in clinical settings.

## Results Overview
Our framework was tested on ECG printouts generated from the PTB-XL dataset. Deep learning models, including EfficientNet and InceptionNet, were trained to classify ECGs as normal or abnormal. Key findings include:
- High classification performance (ROC-AUC up to 0.947 for clean images)
- Robustness of LIME explanations analyzed using Intersection over Union (IoU) metrics
- Image features extracted from explanations helped assess model stability

### Example Visualizations

As the direct output of our tool, user can visualise different types of perturbated imaged and their respective visualisatiosn:

**LIME Explanations for ECG scans with different manipulations:**

![LIME Explanation](./figs/AC_Figure_8.jpg)

The secondart part of the reults consists of statistical report, with 2 key outcomes, model stability report in form of a table and clustering visualisation. 

**IoU scores for different manipulations**

| Id      | Discolored | No metadata | Handwriting | Wrinkles | Average IoU |
|---------|------------|-------------|-------------|----------|-------------|
| Model 1 | 0.163      | 0.305       | 0.283       | 0.207    | 0.240       |
| Model 2 | 0.143      | 0.522       | 0.496       | 0.184    | 0.336       |
| Model 5 | 0.288      | 0.482       | 0.435       | 0.279    | 0.371       |

**Pyradiomics features vs model used**

![Pyradiomics features vs model used](./figs/model_name.jpg)


**Pyradiomics features vs type of manipulation**
![Pyradiomics features vs type of manipulation](./figs/mode.jpg)
