# Re-import necessary libraries as execution state was reset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Generate synthetic data
def generate_data(n_samples, hidden_layer,output_layer, input_dim):
    X = torch.normal(mean=0, std=1, size=(n_samples, input_dim))
    #hidden_layer = torch.randn(input_dim, hidden_dim)
    Z = torch.relu(X @ hidden_layer)
   # output_layer = torch.randn(hidden_dim, 1)
    y = Z @ output_layer
    y = (y - y.mean()) / y.std()
    return X, y

# Define the model
class RandomFeatureNetworkFixed(nn.Module):
    def __init__(self, input_dim, num_features):
        super(RandomFeatureNetworkFixed, self).__init__()
        self.U = torch.randn(num_features, input_dim, device=device)
        self.U = self.U / torch.norm(self.U, dim=1, keepdim=True)
        self.w = nn.Parameter(torch.zeros(num_features, 1, device=device))

    def forward(self, x):
        z = torch.relu(x @ self.U.T)
        out = z @ self.w
        return out.view(-1, 1)

# Full-batch training function
def train_model(model, X_train, y_train, X_test, y_true, optimizer, criterion, epochs):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        if torch.isnan(loss):
            print("NaN loss detected. Skipping this epoch.")
            continue
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_true)
            test_losses.append(test_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

    return train_losses, test_losses

# Experiment setup
input_dim = 32
ground_truth_hidden_dim = 64
teacher_dim = round(ground_truth_hidden_dim * 0.75)
n_samples = 2048
teacher_lr = 3e-3
student_lr = 1e-3
iterations = 4096
student_sizes = [256, 512, 1024, 2048, 4096]

hidden_layer = torch.randn(input_dim, ground_truth_hidden_dim)
output_layer = torch.randn(ground_truth_hidden_dim, 1)
X_train, y_train = generate_data(n_samples=n_samples, hidden_layer=hidden_layer, output_layer=output_layer, input_dim=input_dim)
X_test, y_true = generate_data(n_samples=n_samples, hidden_layer=hidden_layer, output_layer=output_layer, input_dim=input_dim)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_true = y_true.to(device)

# Train teacher model
model_1 = RandomFeatureNetworkFixed(input_dim, teacher_dim).to(device)
print(f"\nTraining Teacher model MTE={teacher_dim}, (Hidden Size = {ground_truth_hidden_dim})")
optimizer_1 = optim.SGD(model_1.parameters(), lr=teacher_lr)
criterion_1 = nn.MSELoss()
train_losses_1, test_losses_1 = train_model(model_1, X_train, y_train, X_test, y_true, optimizer_1, criterion_1, epochs=iterations)

LTE_min_train = np.min(train_losses_1)
LTE_min_test = np.min(test_losses_1)

# Get predicted labels from teacher
model_1.eval()
with torch.no_grad():
    predicted_labels = model_1(X_train)

# Overspecified student training
train_losses = {MST: [] for MST in student_sizes}
test_loss_ratios = {MST: [] for MST in student_sizes}

for MST in student_sizes:
    print(f"\n=== Training Student Model with MST = {MST} ===")
    student_model = RandomFeatureNetworkFixed(input_dim, MST).to(device)
    optimizer_student = optim.SGD(student_model.parameters(), lr=student_lr)
    criterion_student = nn.MSELoss()

    train_losses_student, test_losses_student = train_model(
        student_model, X_train, predicted_labels, X_test, y_true,
        optimizer_student, criterion_student, iterations
    )

    train_losses[MST] = np.array(train_losses_student)
    test_loss_ratios[MST] = np.array(test_losses_student) / (LTE_min_test if LTE_min_test > 0 else 1e-8)

# Plot loss ratios
for split in ["train", "test"]:
    ys = train_losses if split == "train" else test_loss_ratios
    for MST in student_sizes:
        plt.plot(range(iterations), ys[MST], linestyle='-', label=f"MST = {MST}")
    if split == "test":
        plt.ylim(0.7, 1.2)
        plt.ylabel("Loss Ratio")
    else:
        plt.ylabel("Train Losses")
    plt.xlabel("Training Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"wts_relu_{split}.eps") 
    plt.close()
