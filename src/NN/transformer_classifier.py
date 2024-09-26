"""
Brief: this script using multi-head attention to train a transformer model based on PyTorch
        The model is used for classification task on drone tracking dataset
Author: CHEN Yi-xuan
updateDate: 2024-09-24
"""
import os
import numpy as np
import math
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..GNN.GAT_classifier import save_model_with_index

"""
/brief: custom scaler class to conduct normalization on the non-zero values in the data
        but keep the zero values as 0

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: feature_range: the range of the normalized data
"""
class KeepZeroMinMaxScaler:
    def __init__(self, feature_range=(0.1, 1.1)):
        self.feature_range = feature_range

    def fit_transform(self, X):
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])

        non_zero_mask = (X_2d != 0)
        X_scaled = torch.zeros_like(X_2d, dtype=torch.float32)

        for col in range(X_2d.shape[1]):
            col_data = X_2d[:, col]
            if torch.any(non_zero_mask[:, col]):
                col_non_zero = col_data[non_zero_mask[:, col]]
                min_val = col_non_zero.min()
                max_val = col_non_zero.max()
                if min_val != max_val:
                    scaled = (col_non_zero - min_val) / (max_val - min_val)
                    scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
                    X_scaled[non_zero_mask[:, col], col] = scaled

        return X_scaled.reshape(original_shape)


"""
/brief: custom dataset class for radar drone tracking

/author: CHEN Yi-xuan

/date: 2024-09-24
"""
class DroneRadarDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        # normalize the features
        self.Scaler = KeepZeroMinMaxScaler()
        self.features = self.Scaler.fit_transform(self.features)
        # convert to tensor and use float32 type to align with the model weight and bias type
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

"""
/brief: load the data and split it into train, validation, and test datasets
        then create data loaders for future training
        the data contains nan values, so we need to handle it before training in this function

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: data_path: the path of the npy data file
"""
def load_data(data_path):
    # Load the data from object array
    data = np.load(data_path, allow_pickle=True)

    # Separate features and labels
    X = data[:, 0]  # Shape: (58613, 15, 6)
    y = data[:, 1]  # Shape: (58613,)

    # deal with the nan values in the data, convert nan to 0
    for i in range(len(X)):
        X[i] = np.nan_to_num(X[i])

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X.tolist())
    y = torch.LongTensor(y.tolist())

    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create custom datasets
    train_dataset = DroneRadarDataset(X_train, y_train)
    val_dataset = DroneRadarDataset(X_val, y_val)
    test_dataset = DroneRadarDataset(X_test, y_test)

    # Create data loaders
    Batch_Size = 32
    train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)

    return train_loader, val_loader, test_loader


"""
/brief: custom transformer model using multi-head attention

/author: CHEN Yi-xuan

/date: 2024-09-23
"""
class MultiHeadAttentionClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes):
        super(MultiHeadAttentionClassifier, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        # add normalization layer to avoid gradient vanishing
        self.norm = nn.LayerNorm(input_dim)
        hidden_dim = 64
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # create attention mask for zero-paddings
        padding_mask = torch.all(x == 0, dim=-1) # Shape: (batch_size, seq_length)

        x = x.permute(1, 0, 2)  # x shape: (seq_len, batch_size, input_dim)

        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=padding_mask)
        # attn_output = attn_output + 1e-8  # Add small epsilon to avoid division by zero

        attn_output = self.norm(attn_output + x) # Add residual connection and layer normalization

        # Global average pooling
        # pooled = torch.mean(attn_output, dim=0)

        # Compute the mean over valid positions
        valid_counts = (~padding_mask).sum(dim=1).unsqueeze(-1)
        sequence_embedding = attn_output.sum(dim=0) / valid_counts # Shape: (batch_size, input_dim)

        # Fully connected layers
        x = F.leaky_relu(self.fc_input(sequence_embedding), negative_slope=0.01)
        x = self.dropout(x)
        # x = F.relu(self.fc_hidden(x))
        # x = self.dropout(x)
        x = self.fc_output(x)
        x = F.softmax(x, dim=1)

        return x


    def init_weights(self):
        # Initialize the weights with proper values, here using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_normal_(m.in_proj_weight)
                nn.init.xavier_normal_(m.out_proj.weight)
                nn.init.zeros_(m.in_proj_bias)
                nn.init.zeros_(m.out_proj.bias)



"""
/brief: train the transformer model

/author: CHEN Yi-xuan

/date: 2024-09-23

/param: train_loader: the data loader for training
        val_loader: the data loader for validation
        num_epochs: the number of epochs for training
        learning_rate: the learning rate for optimization
        tolerance: the patience for early stopping
"""
def train_model(train_loader, val_loader, num_epochs=30, initial_model = None, learning_rate=1e-4, tolerance=30):
    if initial_model is None:
        # initialize the model
        input_dim = 6
        num_heads = 3
        num_classes = 2
        model = MultiHeadAttentionClassifier(input_dim, num_heads, num_classes)
        model.init_weights()
    else:
        model = initial_model

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # using gradient clipping to avoid gradient explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Early stopping parameters
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    cur_tolerance = tolerance

    best_model = model
    best_combined_loss = float('inf')
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        nan_tag = False
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # load the batch data to the device
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # check isNaN in the input
            # if torch.isnan(batch_features).any():
            #     print("NaN detected in the input data")
            #     nan_tag = True
            # if torch.isnan(batch_labels).any():
            #     print("NaN detected in the input labels")
            #     nan_tag = True


            # Forward pass
            optimizer.zero_grad() # clear the gradients
            outputs = model(batch_features)
            # loss calculation
            loss = criterion(outputs, batch_labels)

            # check loss NaN
            if torch.isnan(loss).any():
                print("NaN detected in the loss")
                print(f"batch_features: {batch_features}")
                print(f"batch_labels: {batch_labels}")
                print(f"outputs: {outputs}")
                print(f"loss: {loss}")
                nan_tag = True

            # loss backward and optimize
            loss.backward()
            optimizer.step()

            # check the model parameters if NaN exists
            if nan_tag:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: weight mean {param.data.mean()}, grad mean {param.grad.mean()}")

            # calculate the loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)


        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)

        # Early stopping
        if epoch > 0:
            if val_loss < best_val_loss or train_loss < best_train_loss:
                cur_tolerance = tolerance
            else:
                cur_tolerance -= 1
            if cur_tolerance == 0:
                print("Early stopping...")
                break
        best_val_loss = min(val_loss, best_val_loss)
        best_train_loss = min(train_loss, best_train_loss)
        combined_loss = math.sqrt(train_loss * val_loss)
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            best_model = model


        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print()

    save_model_with_index(model, "final_model")
    save_model_with_index(best_model, "best_model")

    return model, best_model


"""
/brief: test the model

/author: CHEN Yi-xuan

/date: 2024-09-23

/param: model: the trained model
        test_loader: the data loader for testing
"""
def test_model(model, test_loader):
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == batch_labels).sum().item()

    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data("../../data/event_2/raw_data_padded.npy")
    final_model, best_model = train_model(train_loader, val_loader, num_epochs=200, learning_rate=5e-3)
    test_model(final_model, test_loader)
    print("\n\nBest model:")
    test_model(best_model, test_loader)




