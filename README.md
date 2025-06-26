# IMAGE-CLASSIFICATION-MODEL
COMPANY : CODTECH IT SOLUTIONS

NAME : KARNATI BHAVYA

INTERN ID : CT2MTDM438

DOMAIN : Machine Learning

DURATION : 8 WEEKS

MENTOR : NEELA SANTOSH
# DESCRIPTION
This project implements an Image Classification system using a Convolutional Neural Network (CNN) developed in PyTorch. Image classification is a fundamental problem in computer vision where the goal is to assign a label to an input image from a fixed set of categories. In this case, we use the FashionMNIST dataset — a widely-used benchmark dataset consisting of grayscale images of clothing items, such as shirts, trousers, and shoes.

The goal of this project is to build a functional CNN that can correctly classify images into one of ten predefined clothing categories. The project involves designing the network architecture, training the model, evaluating performance, and visualizing predictions.

# Dataset Information
The dataset used in this project is FashionMNIST, which is available through torchvision.datasets. It contains:

60,000 training images

10,000 test images

Image size: 28x28 pixels (grayscale)

10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot

Each image is labeled with one of these classes, and the data is well-balanced across categories, making it suitable for building and testing a simple CNN model.

# Model Architecture
The CNN used in this project follows a simple and effective architecture:
Convolutional Layer 1 (input: 1x28x28, output: 32 filters of 3x3)
MaxPooling Layer 1 (reduces spatial dimensions)
Convolutional Layer 2 (input: 32x13x13, output: 64 filters of 3x3)
MaxPooling Layer 2
Fully Connected Layer 1 (input: 64*5*5, output: 128 neurons)
Output Layer (10 output units for 10 classes)
The model uses ReLU activation for non-linearity and cross-entropy loss for training, optimized using the Adam optimizer.

# Project Workflow
Importing Libraries:
PyTorch, torchvision, matplotlib, and other necessary modules are imported.

Data Loading and Transformation:
The FashionMNIST dataset is downloaded and transformed into normalized tensors for model consumption. DataLoaders are created for batching during training and testing.

Model Definition:
A custom CNN class is defined inheriting from nn.Module, containing two convolutional layers, max-pooling, and two fully connected layers.

Model Training:
The model is trained over multiple epochs (typically 5 or more), during which loss values are tracked and printed.

Model Evaluation:
The model’s performance is measured on the test dataset. Metrics like accuracy and confusion matrix are calculated. Test predictions are also visualized alongside actual labels.

Model Saving:
The trained model is saved using torch.save() for future reuse or deployment.

Visualization:
Sample predictions and a confusion matrix are plotted using matplotlib and seaborn to visually analyze classification accuracy across all classes.

# Evaluation Metrics
Accuracy Score on test data
Confusion Matrix
Visualization of predictions
(Optional) Loss curves over epochs
The model achieves good performance on the FashionMNIST dataset, showing that even a basic CNN can effectively classify image data.

# Conclusion
This project demonstrates a hands-on implementation of image classification using Convolutional Neural Networks in PyTorch. It covers all the essential steps from data preprocessing to model training, evaluation, and visualization. The results highlight CNN’s effectiveness in handling visual data and its potential use in various real-world applications like product recognition, medical imaging, and face detection.

# OUTPUT

![Image](https://github.com/user-attachments/assets/eeed7074-ea14-4bd2-9f37-22ce6f7d8f72)
