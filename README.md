
Dynamic Weighted Ensemble for Accurate Cervical Cancer Classification in Pap Smear Images
Overview
This project presents a Dynamic Weighted Ensemble method for classifying cancerous and precancerous cells in Pap smear images. The ensemble combines three powerful deep learning models: Swin Transformer, Vision Transformer (ViT), and ResNet50. These models each offer unique strengths, including hierarchical vision processing, global attention mechanisms, and robust feature extraction. By combining these models using Softmax-weighted averaging, we aim to achieve highly accurate classification while reducing false positives.

The method is evaluated on the SipakMed dataset, containing 4,049 images and 966 manually cropped cell clusters. Our ensemble model achieves an impressive accuracy of 97.87%, significantly outperforming many state-of-the-art models in cervical cancer classification.

Key Features
Ensemble of Deep Learning Models: The ensemble integrates Swin Transformer, Vision Transformer (ViT), and ResNet50 to leverage the strengths of each model.
Dynamic Weighted Averaging: Softmax-weighted averaging is used to combine the classification predictions of each model, improving accuracy and reducing false positives.
High Accuracy: Achieved an accuracy of 97.87% on the SipakMed dataset, demonstrating the effectiveness of the approach.
Cervical Cancer Detection: This model assists in early detection of cervical cancer, which is crucial for preventing the disease, especially in remote and underserved areas.
Requirements
To run this project, ensure that the following dependencies are installed:

Python 3.x
TensorFlow 2.x
PyTorch
OpenCV
scikit-learn
Pandas
NumPy
Install the required packages using the following command:

bash
Copy code
pip install -r requirements.txt
Dataset
The model is evaluated on the SipakMed dataset, which consists of:

4,049 images of Pap smear samples.
966 manually cropped cell clusters.
You can download the dataset from SipakMed Dataset.

Usage
1. Clone the Repository
First, clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/cervical-cancer-classification.git
cd cervical-cancer-classification
2. Prepare the Dataset
Download the SipakMed dataset and place it in the data/ directory of the project. Ensure the images are organized in the correct format.

3. Training the Model
To train the ensemble model, use the following command:

bash
Copy code
python train.py
This will load the dataset, preprocess the images, and train the ensemble model on the dataset. It will also save the trained models and logs for future evaluation.

4. Evaluating the Model
To evaluate the trained model on the test set and view the accuracy, use the following command:

bash
Copy code
python evaluate.py
The model's performance, including accuracy and confusion matrix, will be displayed in the console.

Results
The ensemble model achieves an accuracy of 97.87% on the SipakMed dataset, outperforming several existing methods for cervical cancer classification. The dynamic weighted ensemble method successfully combines the strengths of Swin Transformer, ViT, and ResNet50, resulting in robust performance.
