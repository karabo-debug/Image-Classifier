from PIL import Image
from matplotlib import transforms
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random

'''DATA PROCESSING'''
# Load MNIST from file
DATA_DIR = "."
download_dataset = True

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset)

# Store as X_train, y_train, X_test, y_test
X_train = train_mnist.data.float()
y_train = train_mnist.targets
X_test = test_mnist.data.float()
y_test = test_mnist.targets

""" Split training data into training and validation (let validation be the size of test) """

# Sample random indices for validation
test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

# Create validation set
X_valid = X_train[indices]
y_valid = y_train[indices]

# Remove validation set from training set
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

# We need to reshape the data from matrices to vectors
X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Create data loaders
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Apply first linear transformation
        output = self.fc1(x)

        # Apply ReLU activation
        output = self.relu(output)

        # Apply second linear transformation
        output = self.fc2(output)

        return output

""" TRAIN THE MODEL """
from torch.nn import functional as F

# Hyperparameters
num_epochs = 60
learning_rate = 0.001  # Adjusted learning rate
l2_lambda = 0.001  # regularization strength
best_model = None
best_acc = None
best_epoch = None
no_improvement = 5


# Define model, loss function, and optimizer
model = MultinomialLogisticRegression(input_size=28*28, hidden_size=60, num_classes=10)  # Assuming 10 classes for MNIST
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define lists to store training and validation accuracies
train_loss = []
train_accuracy = []
validation_accuracy = []
validation_loss = []


# Training loop with L2 regularization
for epoch in range(num_epochs):
    epoch_loss = []
    epoch_validation_loss = []
    correct_train = 0
    total_train = 0

    # Training phase
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images.view(images.size(0), -1))

        # Compute loss
        loss = criterion(outputs, labels)

        # Calculate L2 regularization term
        l2_reg = sum(torch.norm(param) for param in model.parameters())

        # Add L2 regularization term to the loss
        loss += l2_lambda * l2_reg
        optimizer.zero_grad()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach())

        # Compute training accuracy
        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_loss.append(torch.tensor(epoch_loss).mean())

    # Compute training accuracy for the epoch
    accuracy_train = 100 * correct_train / total_train
    train_accuracy.append(accuracy_train)

    # Validation phase to store validation losses
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.view(images.size(0), -1))
            loss = criterion(outputs, labels)
            epoch_validation_loss.append(loss.item())

    # Compute average validation loss for the epoch
    validation_loss.append(np.mean(epoch_validation_loss))

    # Compute validation accuracy
    total_valid = 0
    correct_valid = 0
    for images, labels in val_loader:
        outputs = model(images.view(images.size(0), -1))
        _, predicted_valid = torch.max(outputs, 1)
        total_valid += labels.size(0)
        correct_valid += (predicted_valid == labels).sum().item()
    accuracy_valid = 100 * correct_valid / total_valid
    validation_accuracy.append(accuracy_valid)

    if best_acc is None or accuracy_valid > best_acc:
        print("New best epoch ", epoch, "acc", accuracy_valid)
        best_acc = accuracy_valid
        best_model = model.state_dict()
        best_epoch = epoch
    if best_epoch + no_improvement <= epoch:
        print("No improvement for", no_improvement, "epochs")
        break

model.load_state_dict(best_model)

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.view(images.size(0), -1))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%\n")

print("Pytorch Output...")
print("...")
print("Done!\n")

# Allow the user to input file paths and perform inference
while True:
    file_path = input("Please enter a filepath (or 'exit' to quit): ")
    if file_path.lower() == 'exit':
        print("Exiting...")
        break
    
    try:
        # Read the image using PIL
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize the image to match model input size
            transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Perform inference using the trained model
        with torch.no_grad():
            model.eval()
            output = model(image.view(1,-1))  # Flatten the image
        
            # Get the predicted label
            predicted = torch.argmax(output, dim=1).item()
           
            # Display the predicted label
            print("Classifier:", predicted)
        
    except Exception as e:
        print("Error:", e)