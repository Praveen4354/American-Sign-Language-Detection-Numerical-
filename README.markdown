# American Sign Language Detection Project

## Overview
This project focuses on detecting and classifying American Sign Language (ASL) letters from images using deep learning models. The repository includes scripts to create a custom dataset, train multiple machine learning models, evaluate their performance, and use pre-trained models for inference. The goal is to accurately recognize ASL gestures to facilitate communication.

## Dataset
The dataset used for this project is the [American Sign Language Dataset (Numerical)](https://www.kaggle.com/datasets/praveen1910/american-sign-language-dataset-numerical) available on Kaggle. It contains images of ASL letters represented numerically, suitable for training machine learning models.

## Repository Structure
The repository contains the following files and directories:

- `create_dataset.py`: Script to generate a custom dataset for ASL detection.
- `train_keras_model.py`: Script to train and save a Keras-based CNN model.
- `train_vgg_model.py`: Script to train and save a VGG-based model.
- `train_vit_model.py`: Script to train and save a Vision Transformer (ViT) model.
- `train_elu_cnn_model.py`: Script to train and save an ELU-based CNN model.
- `saved_models/`: Directory containing pre-trained model weights:
  - `keras_model.h5`: Saved Keras CNN model.
  - `vgg_model.h5`: Saved VGG model.
  - `vit_model.h5`: Saved Vision Transformer model.
  - `elu_cnn_model.h5`: Saved ELU-based CNN model.
- `knn_evaluation.py`: Script to evaluate the dataset using a K-Nearest Neighbors (KNN) algorithm.
- `evaluate_models.py`: Script to evaluate the performance metrics of the trained models.
- `README.md`: This file, providing an overview and instructions for the project.

## Prerequisites
To run the scripts in this repository, ensure you have the following installed:
- Python 3.8 or higher
- Required Python libraries (install via `pip`):
  ```bash
  pip install numpy pandas tensorflow opencv-python scikit-learn matplotlib
  ```
- Access to the dataset from [Kaggle](https://www.kaggle.com/datasets/praveen1910/american-sign-language-dataset-numerical). Download and extract it to a local directory.
- (Optional) GPU support for faster model training (requires compatible hardware and TensorFlow GPU setup).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file is not provided, manually install the libraries listed in the Prerequisites section.

3. **Download the Dataset**:
   - Download the ASL dataset from [Kaggle](https://www.kaggle.com/datasets/praveen1910/american-sign-language-dataset-numerical).
   - Extract the dataset to a folder (e.g., `data/`) in the repository root.
   - Update the dataset path in `create_dataset.py` and other scripts if necessary.

## Usage
Follow these steps to use the repository:

1. **Create a Custom Dataset** (Optional):
   - Run the `create_dataset.py` script to generate a custom dataset if needed:
     ```bash
     python create_dataset.py
     ```
   - Ensure the script points to the correct dataset directory or modify it to suit your needs.

2. **Train Models**:
   - Train individual models using the respective scripts. For example:
     ```bash
     python train_keras_model.py
     python train_vgg_model.py
     python train_vit_model.py
     python train_elu_cnn_model.py
     ```
   - Each script trains a model and saves it to the `saved_models/` directory.
   - Ensure the dataset path in each script is correctly set to the downloaded or custom dataset.

3. **Evaluate Models**:
   - Use the `evaluate_models.py` script to compute performance metrics (e.g., accuracy, precision, recall) for the trained models:
     ```bash
     python evaluate_models.py
     ```
   - The script loads the saved models from `saved_models/` and outputs metrics to the console or a file.

4. **KNN Evaluation**:
   - Run the `knn_evaluation.py` script to evaluate the dataset using a KNN classifier:
     ```bash
     python knn_evaluation.py
     ```
   - This provides a baseline comparison for the deep learning models.

5. **Inference with Pre-trained Models**:
   - Use the saved models (`keras_model.h5`, `vgg_model.h5`, `vit_model.h5`, `elu_cnn_model.h5`) for inference on new images.
   - Example (modify paths as needed):
     ```python
     from tensorflow.keras.models import load_model
     import cv2
     import numpy as np

     model = load_model('saved_models/keras_model.h5')
     image = cv2.imread('path_to_image.jpg')
     # Preprocess image (resize, normalize, etc.)
     prediction = model.predict(image)
     print(prediction)
     ```

## Results
The trained models (Keras, VGG, ViT, ELU-CNN) and KNN evaluation provide different performance metrics. Refer to the output of `evaluate_models.py` for detailed results, including accuracy, precision, recall, and F1-score for each model.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open an issue on the repository or contact the maintainer at [your-email@example.com].