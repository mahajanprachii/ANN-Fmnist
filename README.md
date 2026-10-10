# 👕 Fashion-MNIST Classification Using an Artificial Neural Network (ANN)

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** using PyTorch to classify images from the Fashion-MNIST dataset into 10 different categories of clothing and fashion items.

A smaller subset of the dataset was used for training to understand the fundamentals of deep learning, including data preprocessing, custom dataset creation, model training, and evaluation.

## 📊 Dataset

- **Dataset:** Fashion-MNIST
- **Total samples used:** 6,000 images (a reduced subset of the original dataset).
- **Image dimensions:** 28 × 28 pixels
- **Number of classes:** 10

Each image was flattened into a vector of 784 features before being passed to the fully connected neural network.


## ⚙️ Data Preprocessing

### 1. Pixel Normalization

Pixel values were divided by `255.0` to scale them from the range **0–255 to 0–1**.

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

This normalization helps provide appropriately scaled inputs to the neural network.

### 2. Custom Dataset Class

A custom dataset class was implemented by inheriting from PyTorch's `Dataset` class.

- `__init__()` — stores the input features and labels.
- `__len__()` — returns the total number of samples.
- `__getitem__()` — returns a sample and its corresponding label.

### 3. DataLoader

PyTorch's `DataLoader` was used to load the dataset in batches, making the training process more manageable and efficient.

## 🧠 Model Architecture

A fully connected feedforward ANN was implemented using PyTorch.

| Layer | Configuration |
|---|---|
| Input layer | 784 features |
| Hidden layer 1 | 128 neurons |
| Hidden layer 2 | 64 neurons |
| Output layer | 10 neurons |

The output layer generates scores for the 10 possible classes, and the class with the highest score is selected as the prediction.

## 🚀 Model Training

The model was trained using the following configuration:

- **Framework:** PyTorch
- **Loss function:** Cross Entropy Loss
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Number of epochs:** 100

During training, the model performed a forward pass, calculated the loss, computed gradients using backpropagation, and updated its weights using the optimizer.

### Training Result

Final reported average training loss after 100 epochs:

```text
0.000083106
```

## 📈 Model Evaluation

After training, the model was switched to evaluation mode using:

```python
model.eval()
```

Gradient tracking was disabled using `torch.no_grad()` because gradients and weight updates were not required during evaluation.

The model was evaluated on the test dataset by comparing predicted classes with the actual labels.

### Evaluation Result

- **Test Accuracy: 83.25%**

The model correctly classified approximately 83.25% of the evaluated test samples.

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pandas

## 🎯 Key Learnings

- Understanding artificial neural networks and fully connected layers.
- Implementing a custom PyTorch Dataset class.
- Loading
