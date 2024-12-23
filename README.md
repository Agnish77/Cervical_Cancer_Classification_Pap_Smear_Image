# Dynamic Weighted Ensemble for Accurate Cervical Cancer Classification in Pap Smear Images

# Overview
This project presents a Dynamic Weighted Ensemble method for classifying cancerous and precancerous cells in Pap smear images. The ensemble combines three powerful deep learning models: Swin Transformer, Vision 

Transformer (ViT), and ResNet50. These models each offer unique strengths, including hierarchical vision processing, global attention mechanisms, and robust feature extraction. By combining these models using Softmax-

weighted averaging, we aim to achieve highly accurate classification while reducing false positives.

The method is evaluated on the SipakMed dataset, containing 4,049 images and 966 manually cropped cell clusters. Our ensemble model achieves an impressive accuracy of 97.87%, significantly outperforming many state-of-the-

art models in cervical cancer classification.

## Key Features
Ensemble of Deep Learning Models: The ensemble integrates Swin Transformer, Vision Transformer (ViT), and ResNet50 to leverage the strengths of each model.

Dynamic Weighted Averaging: Softmax-weighted averaging is used to combine the classification predictions of each model, improving accuracy and reducing false positives.

High Accuracy: Achieved an accuracy of 97.87% on the SipakMed dataset, demonstrating the effectiveness of the approach.

Cervical Cancer Detection: This model assists in early detection of cervical cancer, which is crucial for preventing the disease, especially in remote and underserved areas.

The model is evaluated on the SipakMed dataset, which consists of:

4,049 images of Pap smear samples.

966 manually cropped cell clusters.

You can download the dataset from https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed



Results
The ensemble model achieves an accuracy of 95.22% on the SipakMed dataset with a train-test split ratio of 70-30, outperforming several existing methods for cervical cancer classification. The dynamic weighted ensemble method successfully combines the strengths of Swin Transformer, ViT, and ResNet50, resulting in robust performance.
