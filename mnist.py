import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transform)

# Split dataset for teacher (model_1) and student (model_2)
train_size_1 = len(train_dataset) // 2
train_size_2 = len(train_dataset) - train_size_1
train_dataset_1, train_dataset_2 = random_split(train_dataset, [train_size_1, train_size_2])

train_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=False)
train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Teacher sizes to test and fixed student size
hidden_dims_model_1 = [50, 75, 100]
hidden_dim_model_2 = 300
criterion = nn.CrossEntropyLoss()

# Accuracies
hidden_dim_accuracies_model_1 = []
hidden_dim_accuracies_model_1_model_2 = []

for hidden_dim_model_1 in hidden_dims_model_1:
    print(f"\nTraining model_1 with hidden_dim = {hidden_dim_model_1}")

    model_1 = NeuralNet(input_size, hidden_dim_model_1, num_classes).to(device)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

    # Train model_1
    for epoch in range(num_epochs):
        model_1.train()
        for images, labels in train_loader_1:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model_1(images)
            loss = criterion(outputs, labels)
            optimizer_1.zero_grad()
            loss.backward()
            optimizer_1.step()

    # Evaluate model_1
    model_1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model_1(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_model_1 = 100 * correct / total
    hidden_dim_accuracies_model_1.append((hidden_dim_model_1, accuracy_model_1))

    # Generate predicted labels for student training
    predicted_labels = []
    model_1.eval()
    with torch.no_grad():
        for images, _ in train_loader_1:
            images = images.reshape(-1, 28*28).to(device)
            outputs = model_1(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())

    predicted_labels = torch.tensor(predicted_labels, dtype=torch.long)
    original_data = train_dataset.data[train_dataset_1.indices]
    dataset_with_predicted_labels = TensorDataset(original_data.view(-1, 28*28).float() / 255.0, predicted_labels)
    combined_dataset = ConcatDataset([dataset_with_predicted_labels, TensorDataset(
        train_dataset.data[train_dataset_2.indices].view(-1, 28*28).float() / 255.0,
        train_dataset.targets[train_dataset_2.indices]
    )])
    train_loader_with_predicted_labels = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

    # Train and evaluate model_2 for each teacher
    model_2 = NeuralNet(input_size, hidden_dim_model_2, num_classes).to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
    epoch_accuracies = []

    for epoch in range(num_epochs):
        model_2.train()
        for images, labels in train_loader_with_predicted_labels:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model_2(images)
            loss = criterion(outputs, labels)
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()

        # Evaluate model_2 on test set
        model_2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model_2(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy_model_2 = 100 * correct / total
        epoch_accuracies.append(accuracy_model_2)
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy of model_2: {accuracy_model_2}%")

    mean_accuracy = np.mean(epoch_accuracies)
    std_accuracy = np.std(epoch_accuracies)
    hidden_dim_accuracies_model_1_model_2.append((hidden_dim_model_1, mean_accuracy, std_accuracy))
    print(f"\nMean Accuracy of model_2 with model_1 hidden_dim = {hidden_dim_model_1}: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

# Summary of results
print("\nFinal comparison of model_2 accuracies for different model_1 hidden_dims:")
for hidden_dim_model_1, mean_accuracy, std_accuracy in hidden_dim_accuracies_model_1_model_2:
    print(f"Model_1 Hidden Dim: {hidden_dim_model_1}, Model_2 Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
