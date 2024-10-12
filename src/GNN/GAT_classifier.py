"""
Brief: This script implement a radar track classifier for drone-discrimination
        using the Graph Attention Network (GAT)

Author: CHEN Yi-xuan

updateDate: 2024-09-27
"""
import os
import math
import numpy as np
import torch
from torch.distributed.pipeline.sync.checkpoint import checkpoint
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

"""
/brief: custom dataset for drone radar track data used in the GNN model
        get the data from the npy file and convert it to PyTorch Geometric Data objects

/author: CHEN Yi-xuan

/date: 2024-09-26
"""
class DroneRadarDataset(Dataset):
    def __init__(self, data_path):
        super(DroneRadarDataset, self).__init__()
        self.data = np.load(data_path, allow_pickle=True)

    def len(self):
        return len(self.data)

    def get(self, idx):
        # Each item in self.data is a tuple (track, label)
        track, label, _ = self.data[idx]

        # deal with the nan values in the data, convert nan to 0
        track = np.array(track)
        track = np.nan_to_num(track)

        # expand a label list having same len with track to let each point in the track have a label
        expanded_label = np.array([label] * len(track))

        # Create edge index
        num_nodes = track.shape[0]
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])  # Connect to next node
            if i < num_nodes - 2:
                edge_index.append([i, i + 2])  # Connect to node two steps ahead

        # Scale the node features to avoid numerical instability
        scaler = MinMaxScaler(feature_range=(0.1, 1.1))
        x = scaler.fit_transform(track)

        # Convert to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        y = torch.tensor(expanded_label, dtype=torch.long)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        return data


"""
/brief: load data from the dataset and split it into train and test sets

/author: CHEN Yi-xuan

/date: 2024-09-26

/param: data_path: str, path to the npy file containing the dataset
        batch_size: int, batch size for the data loaders
        test_size: float, proportion of the dataset to include in the test set
        random_state: int, random seed for reproducibility
"""
def load_data(data_path, batch_size=32, test_size=0.2, random_state=42):
    # Construct graphs and create PyTorch Geometric dataset
    dataset = DroneRadarDataset(data_path)

    # Print some information about the dataset
    print(f"Dataset contains {len(dataset)} samples")
    print("using the first track as an example:")
    sample = dataset[0]
    print(f"Sample graph has {sample.num_nodes} nodes and {sample.num_edges} edges")
    print(f"Node feature shape: {sample.x.shape}")
    print(f"Label: {sample.y}")

    # Split dataset into train and test
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=random_state
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


"""
/brief: Graph Attention Network (GAT) for node(point in radar track) classification

/author: CHEN Yi-xuan

/date: 2024-09-26
"""
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=3):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.6)
        self.gat3 = GATConv(hidden_dim * num_heads, num_classes, heads=1, dropout=0.6)
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat3(x, edge_index))

        # x = torch.mean(x, 0)  # Global mean pooling
        # x = self.lin(x)

        x = F.log_softmax(x, dim=1) # Log softmax for classification

        return x


"""
/brief: small tool to save the model with index to avoid overwriting

/author: CHEN Yi-xuan

/date: 2024-09-26

/param: model: the model to be saved
        base_filename: the base filename(description) for the model
"""
def save_model_with_index(model, base_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index = 1
    while True:
        filename = f"{base_filename}_{index}.pth"
        full_path = os.path.join(script_dir, filename)

        if not os.path.exists(full_path):
            torch.save(model, full_path)
            print(f"Model saved as: {filename}")
            break

        index += 1

        return full_path


"""
/brief: train the transformer model

/author: CHEN Yi-xuan

/date: 2024-09-27

/param: train_loader: the data loader for training
        val_loader: the data loader for validation
        num_epochs: the number of epochs for training
        learning_rate: the learning rate for optimization
        tolerance: the patience for early stopping
"""
def train_model(train_loader, test_loader, initial_model = None, num_epochs=200, learning_rate=0.005, tolerance=30):
    # Initialize model
    if initial_model is None:
        model = GAT(input_dim=6, hidden_dim=16, num_classes=2)
    else:
        model = initial_model

    # Train and evaluate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10.0)

    # initialize the model for training
    model.train()

    # Early stopping parameters
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_combined_acc = 0.0
    best_model = model
    patience = tolerance

    # set up checkpoint for the model in case of breaking down or for further training
    checkpoint_path = save_model_with_index(model, "checkpoint")

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0.0 # Accumulate the training loss
        train_correct = 0 # Accumulate the number of correct predictions
        train_len = 0 # node-classification, accumulate the number of nodes
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            optimizer.zero_grad() # Clear gradients
            output = model(data) # Forward pass
            loss = F.nll_loss(output, data.y) # Calculate loss
            # Backward pass
            loss.backward()
            optimizer.step()
            # Accumulate the training loss
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += int((pred == data.y).sum())
            train_len += data.y.shape[0]

        # update the checkpoint after each epoch
        torch.save(model, checkpoint_path)

        # Calculate the training accuracy
        train_acc = train_correct / train_len
        train_loss /= len(train_loader)

        # Evaluate the model on the test set
        test_correct = 0
        test_loss = 0.0
        test_len = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = model(data)
                loss = F.nll_loss(output, data.y)
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_correct += int((pred == data.y).sum())
                test_len += data.y.shape[0]

        # Calculate the test accuracy
        test_acc = test_correct / test_len
        test_loss /= len(test_loader)

        # Print the training and test accuracy
        print(f"Epoch {epoch+1}/{num_epochs})")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Early stopping
        combined_acc = math.sqrt(train_acc * test_acc)
        if train_acc > best_train_acc or test_acc > best_test_acc:
            patience = tolerance
        else:
            patience -= 1
        if patience == 0:
            print("Early stopping...")
            break
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)

        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            best_model = model

    # save the best model and final model under the same directory of the script
    save_model_with_index(model,  "final_model")
    save_model_with_index(best_model, "best_model")

    # the training process is done, remove the checkpoint
    os.remove(checkpoint_path)

    return model, best_model


"""
/brief: test the model

/author: CHEN Yi-xuan

/date: 2024-09-26

/param: model: the trained model
        test_loader: the data loader for testing
"""
def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    correct = 0
    test_len = 0.0
    for data in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += int((pred == data.y).sum())
        test_len += data.y.shape[0]
    print(f"Test Accuracy: {correct / test_len}")
    return correct / test_len


if __name__ == '__main__':
    data_path = '../../data/event_2/raw_tracks_graph.npy'
    train_loader, test_loader = load_data(data_path)
    # model, best_model = train_model(train_loader, test_loader, learning_rate=1e-3)

    best_model = torch.load("./best_model_1.pth")
    test_model(best_model, test_loader)
