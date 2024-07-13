Sure! Here is the updated `README.md` file with an additional section comparing the EfficientNetB2 model with a custom CNN model, including placeholders for graphs:

---

<p align="center">
  <img src="M:\Projects\machineLearning\Plant_Disease_Prediction\home_page.jpg" alt="VerdantVision">
</p>

# VerdantVision

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-brightgreen)

VerdantVision is a WebApp designed to predict and monitor the health and condition of plants, focusing on their lush, green, and healthy state. It leverages the state-of-the-art EfficientNetB2 architecture for accurate classification and prediction.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Model Testing](#model-testing)
- [Data Augmentation](#data-augmentation)
- [Model Comparison](#model-comparison)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Introduction
VerdantVision uses machine learning to analyze images of plants and predict their health status. The core of the application is powered by EfficientNetv2B0, which is known for its balance between accuracy and computational efficiency. The application is built using Streamlit, allowing for an interactive and user-friendly web interface.

## Features
- Predicts the health status of plants using image data.
- Utilizes EfficientNetB2 for high accuracy and efficiency.
- Easy-to-use interface for uploading and analyzing plant images.
- Data augmentation for improving model robustness.
- A Custom CNN Model has been used for comparison of performance between EfficientNetv2B0 and the Custom Model

## Installation
### Prerequisites
- Python 3.7+
- pip (Python package installer)

### EfficientNetV2B0 Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/Koustavdas0423/VerdantVision.git
    cd VerdantVision
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the training and test scripts(without web app Usage):
    ```sh
    python plant_disease_prediction_efficient_net_v2B0.py 
    python test.py
    ```

## For App Usage
1. Run the Streamlit app:
    ```sh
    streamlit run main.py
    ```

2. Follow the instructions on the web interface to upload plant images and receive health predictions.

### Custom CNN Model Usage
The CNN Test And Train scipt is provided in the the same directory:
```sh
jupyter Train_plant_disease.ipynb
jupyter Test_plant_disease.ipynb
```

## Dataset
We use the publicly available and widely recognized PlantVillage Dataset. The dataset was published by crowdAI during the "PlantVillage Disease Classification Challenge."

The dataset consists of about 54,305 images of plant leaves collected under controlled environmental conditions. Place the dataset directories correctly in the scipts The plant images span the following 14 species:

| Species         | Classes                                                   |
|-----------------|-----------------------------------------------------------|
| Apple           | Apple Scab, Apple Black Rot, Apple Cedar Rust, Apple Healthy |
| Blueberry       | Blueberry Healthy                                         |
| Cherry          | Cherry Healthy, Cherry Powdery Mildew                     |
| Corn            | Corn Northern Leaf Blight, Corn Gray Leaf Spot, Corn Common Rust, Corn Healthy |
| Grape           | Grape Black Rot, Grape Black Measles, Grape Leaf Blight, Grape Healthy |
| Orange          | Orange Huanglongbing                                      |
| Peach           | Peach Bacterial Spot, Peach Healthy                       |
| Bell Pepper     | Bell Pepper Healthy, Bell Pepper Bacterial Spot           |
| Potato          | Potato Early Blight, Potato Healthy, Potato Late Blight   |
| Raspberry       | Raspberry Healthy                                         |
| Soybean         | Soybean Healthy                                           |
| Squash          | Squash Powdery Mildew                                     |
| Strawberry      | Strawberry Healthy, Strawberry Leaf Scorch                |
| Tomato          | Tomato Bacterial Spot, Tomato Early Blight, Tomato Late Blight, Tomato Leaf Mold, Tomato Septoria Leaf Spot, Tomato Two Spotted Spider Mite, Tomato Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus, Tomato Healthy |

Due to the limited computational power, it is difficult to train the classification model locally on a majority of normal machines. Therefore, we use the processing power offered by Google Colab notebook as it connects us to a free TPU instance quickly and effortlessly.

## For EfficientNetV2B0 Model Training
The model training process is outlined in the `plant_disease_prediction_efficient_net_v2B0.py ` file. This notebook includes data loading, preprocessing, augmentation, model training, and evaluation steps.

## Model Performance
The performance of the EfficientNetv2B0 model during training and validation is as follows:

### Training, Validation Accuracy and Loss
![Training Accuracy and Loss](M:\Projects\machineLearning\Plant_Disease_Prediction\images\accuracy_and_loss.jpg)

### Summary of Results
- **Training Accuracy:** 98.83%
- **Validation Accuracy:** 97.86%
- **Training Loss:** 0.034
- **Validation Loss:** 0.065

## Model Testing
The model testing and evaluation are detailed in the `test.py` file. This includes loading the trained model, running it on a test dataset, and visualizing the results.

### Confusion Matrix
![Confusion Matrix](M:\Projects\machineLearning\Plant_Disease_Prediction\images\conf_matrix.jpg)

### Sample Predictions
![Sample Predictions](M:\Projects\machineLearning\Plant_Disease_Prediction\PotatoEarlyBlight2.jpg)

## Data Augmentation
Data augmentation is crucial for improving the model's robustness and generalization ability. The augmented dataset is stored in the `augmented_dataset/` directory.

### Data Augmentation Techniques
- Rotation
- Flipping
- Scaling
- Color Jittering

## For EfficientNetV2B0 Model Training
The model training process is outlined in the `Train_plant_disease.ipnyb ` file. This notebook includes data loading, preprocessing, augmentation, model training, and evaluation steps.

To leverage Google Colab for training, follow these steps:
1. Upload your training notebook (`Train_plant_disease.ipnyb`) to Google Colab.
2. Ensure you have a TPU instance enabled: `Runtime` -> `Change runtime type` -> Select `TPU` in the `Hardware accelerator` dropdown.
3. Execute the notebook cells to train the model.

## Model Performance
The performance of the EfficientNetv2B0 model during training and validation is as follows:

### Training, Validation Accuracy and Loss
![Training Accuracy and Loss](M:\Projects\machineLearning\Plant_Disease_Prediction\Cnn_acc_val.jpg)

### Summary of Results
- **Training Accuracy:** 98.83%
- **Validation Accuracy:** 97.86%
- **Training Loss:** 0.034
- **Validation Loss:** 0.065

## Model Testing
The model testing and evaluation are detailed in the `Test_plant_disease.ipnyb` file. This includes loading the trained model, running it on a test dataset, and visualizing the results.

### Confusion Matrix
![Confusion Matrix](M:\Projects\machineLearning\Plant_Disease_Prediction\Cnn_conf_matrix.jpg)

## Model Comparison
To evaluate the performance of EfficientNetB2, we compared it with a custom Convolutional Neural Network (CNN) model.

### Custom CNN Model Architecture
The custom CNN model consists of the following layers:
- Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer 1: 2x2 pool size
- Strides layer 1: 2
- Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer 2: 2x2 pool size
- Strides layer 2: 2
- Fully Connected Layer 3: 128 units, ReLU activation
- Max Pooling Layer 3: 2x2 pool size
- Strides layer 3: 2
- Fully Connected Layer 4: 256 units, ReLU activation
- Max Pooling Layer 4: 2x2 pool size
- Strides layer 4: 2
- Fully Connected Layer 5: 512 units, ReLU activation
- Max Pooling Layer 5: 2x2 pool size
- Strides layer 5: 2
- Droput
- Flatten
- Dense
- Output Layer: Softmax activation

### Performance Comparison
We trained both models on the same dataset and compared their performance in terms of accuracy and loss.

### EfficientNetV2b0 Training, Validation Accuracy and Loss
![Training Accuracy and Loss](M:\Projects\machineLearning\Plant_Disease_Prediction\images\accuracy_and_loss.jpg)

### CNN Model Training, Validation Accuracy and Loss
![Training Accuracy and Loss](M:\Projects\machineLearning\Plant_Disease_Prediction\Cnn_acc_val.jpg)

### Summary of Results
- **EfficientNetV2B0 Training Accuracy:** 98.83%
- **EfficientNetV2b0 Validation Accuracy:** 97.86%
- **Custom CNN Training Accuracy:** 97.81%
- **Custom CNN Validation Accuracy:** 94.45%
- **EfficientNetB2 Training Loss:** 0.034
- **EfficientNetB2 Validation Loss:** 0.065
- **Custom CNN Training Loss:** 0.078
- **Custom CNN Validation Loss:** 0.249

## References
This project leverages research and methodologies from several key papers and resources:
- Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. *arXiv preprint arXiv:2104.00298*. [Link to paper](https://arxiv.org/abs/2104.00298)
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. [Link to paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
- Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv preprint arXiv:1704.04861*. [Link to paper](https://arxiv.org/abs/1704.04861)

## Contributing
Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are reviewed on a regular basis.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

