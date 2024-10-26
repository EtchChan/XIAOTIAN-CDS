"""
Brief: This script implement a radar track classifier for drone-discrimination
        using the Graph Attention Network (GAT)

Author: CHEN Yi-xuan

updateDate: 2024-10-13
"""
import os
import math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.data_preprocess import extract_event_2_data_from_csv

def xlsx_to_csv(xlsx_path):
    data_xlsx = pd.read_excel(xlsx_path, index_col=0)
    csv_path = xlsx_path.replace('xlsx', 'csv')
    data_xlsx.to_csv(csv_path, encoding='utf-8')
    return csv_path


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
        # Each item in self.data is a tuple (track, label, len)
        track, label, track_len = self.data[idx]

        # deal with the nan values in the data, convert nan to 0
        track = np.array(track)
        track = np.nan_to_num(track)

        # expand a label list having same len with track to let each point in the track have a label
        expanded_label = np.array([label] * len(track))

        # Create edge index
        num_nodes = track.shape[0]
        edge_index = []
        step_range = 5
        for i in range(num_nodes - 1):
            for step in range(1, step_range):  # Set the window size for the edges to connect as 5
                if i + step < num_nodes:  # Check if the target node exists
                    edge_index.append([i, i + step])

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

    if test_size < 1.0:
        # Split dataset into train and test
        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            test_size=test_size,
            random_state=random_state
        )
    else:
        # Use the entire dataset for testing
        train_indices = []
        test_indices = list(range(len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    if test_size < 1.0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = []
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
    # Use the current working directory instead of the script directory
    current_dir = os.getcwd()
    index = 1
    while True:
        filename = f"{base_filename}_{index}.pth"
        full_path = os.path.join(current_dir, filename)
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
    checkpoint_path = str(save_model_with_index(model, "checkpoint"))

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
        borrow_checkpoint_path = checkpoint_path
        torch.save(model, borrow_checkpoint_path)

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
def test_model(model_list, test_loader, weight_list=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # model.eval()
    # correct = 0
    # test_len = 0.0
    # for data in tqdm(test_loader, desc="Testing"):
    #     data = data.to(device)
    #     output = model(data)
    #     pred = output.argmax(dim=1)
    #     correct += int((pred == data.y).sum())
    #     test_len += data.y.shape[0]
    # print(f"Test Accuracy(Point by Point): {correct / test_len}")

    if weight_list is None:
        weight_list = [1.0, 1.0, 1.0]

    all_graph_preds = []
    all_graph_labels = []

    for idx, model in enumerate(model_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Evaluate the model on the test set
        graph_preds = []
        graph_labels = []
        for data in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            output = model(data)
            node_preds = output.argmax(dim=1)

            for i in range(data.num_graphs):
                mask = data.batch == i
                graph_pred = node_preds[mask].float().mean().item()
                graph_label = data.y[mask].float().mean().item()

                graph_preds.append(graph_pred * weight_list[idx] * len(model_list))
                graph_labels.append(graph_label)

        # append the graph-level predictions and labels of one model to the list
        all_graph_preds.extend(graph_preds)
        all_graph_labels.extend(graph_labels)

    # calculate the final graph-level predictions and labels by simple voting
    all_graph_preds = np.array(all_graph_preds).reshape(-1, len(model_list))
    all_graph_labels = np.array(all_graph_labels).reshape(-1, len(model_list))
    all_graph_preds = np.mean(all_graph_preds, axis=1)
    all_graph_preds = np.where(all_graph_preds >= 0.5, 1, 0)
    all_graph_labels = np.mean(all_graph_labels, axis=1)
    all_graph_labels = np.where(all_graph_labels >= 0.5, 1, 0)

    # Calculate metrics
    accuracy = (all_graph_preds == all_graph_labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_graph_labels, all_graph_preds, average='binary')

    print(f"Graph-level Test Accuracy: {accuracy:.4f}")
    print(f"Graph-level Precision: {precision:.4f}")
    print(f"Graph-level Recall: {recall:.4f}")
    print(f"Graph-level F1 Score: {f1:.4f}")

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_graph_labels, all_graph_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Event 2: Graph-level Confusion Matrix of GAT classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("./GAT_classifier_confusion_matrix.png")
    plt.show()

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Event 2: Normalized Graph-level Confusion Matrix of GAT classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("./GAT_classifier_normalized_confusion_matrix.png")
    plt.show()

    return accuracy, precision, recall, f1


def predict(model_list, data_loader, weight_list=None):
    """
    /brief: predict the labels of the radar track data and append the predicted labels to the data
            the input data's label is its track index instead of the label
            the predicted label will be appended to the data as the last column after the track index
    """

    if weight_list is None:
        weight_list = [1.0, 1.0, 1.0]

    all_graph_preds = []

    for idx, model in enumerate(model_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        graph_preds = []
        for data in tqdm(data_loader, desc="Predicting"):
            data = data.to(device)
            output = model(data)
            node_preds = output.argmax(dim=1)

            for i in range(data.num_graphs):
                mask = data.batch == i
                graph_pred = node_preds[mask].float().mean().item()

                graph_preds.append(graph_pred * weight_list[idx] * len(model_list))

        all_graph_preds.extend(graph_preds)

    # Convert to numpy arrays
    all_graph_preds = np.array(all_graph_preds).reshape(-1, len(model_list))
    all_graph_preds = np.mean(all_graph_preds, axis=1)
    all_graph_preds = np.where(all_graph_preds >= 0.5, 1, 0)

    # examine if the predictions are all zeros, if they are, print invalid results
    if np.sum(all_graph_preds) == 0:
        print("\n!!!!!!!!\nInvalid predictions! all zeros.\n!!!!!!!!\n")

    return all_graph_preds


def append_predictions(input_csv_path, predictions):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Add a new column for predictions, initially filled with empty values
    df['预测标签'] = ''

    # Keep track of the prediction index
    pred_idx = 0

    # Keep track of whether we're in a new track
    last_label_was_empty = True

    # Iterate through the rows
    for i in range(len(df)):
        # Check if this row indexed (non-empty in the '航迹序号' column)
        current_label = str(df.iloc[i]['航迹序号']).strip()
        has_label = current_label != '' and current_label != 'nan'

        if has_label and last_label_was_empty:
            # This is the first row of a new track
            # Add the prediction here
            if pred_idx < len(predictions):
                df.at[i, '预测标签'] = predictions[pred_idx]
                pred_idx += 1

        last_label_was_empty = not has_label

    output_path = input_csv_path.replace('.csv', '_with_predictions.csv')
    # Save the modified dataframe to a new CSV file
    df.to_csv(output_path, index=False)


"""
/brief: merge drone data(label 1) from the preliminary dataset to finals dataset for fine-tuning
        cause there are too few drone instances in the finals dataset
"""
def construct_merged_dataset(preliminary_data_path, finals_data_path):
    # Read the npy files
    preliminary_data = np.load(preliminary_data_path, allow_pickle=True)
    finals_data = np.load(finals_data_path, allow_pickle=True)

    # Extract the drone data from the preliminary dataset
    drone_data = []
    for track, label, track_len in preliminary_data:
        if label == 1:
            drone_data.append((track, label, track_len))

    # Merge the drone data with the finals dataset
    merged_data = np.concatenate((finals_data, drone_data), axis=0)

    # Save the merged dataset to a new npy file
    merged_data_path = finals_data_path.replace('.npy', '_merged.npy')
    np.save(merged_data_path, merged_data)

    return merged_data_path


if __name__ == '__main__':
    # # Pretrain the model and validate it
    # data_path = '../../data/event_2/Train_Preliminary.csv'
    # npy_path = extract_event_2_data_from_csv(data_path)
    # train_loader, test_loader = load_data(npy_path, test_size=0.1)
    # _, best_model = train_model(train_loader, test_loader, learning_rate=1e-3, tolerance=20)
    #
    # # Fine-tune the model with the finals dataset
    # data_path = '../../data/event_2/Train_Finals.csv'
    # npy_path = extract_event_2_data_from_csv(data_path)
    # train_loader, test_loader = load_data(npy_path, test_size=0.05)
    # _, best_model = train_model(train_loader, test_loader, initial_model=best_model, learning_rate=5e-4, tolerance=20)

    # fine-tune the model with the merged dataset
    # preliminary_data_path = '../../data/event_2/Train_Preliminary_tracks_graph.npy'
    # finals_data_path = '../../data/event_2/Train_Finals_tracks_graph.npy'
    # merged_data_path = construct_merged_dataset(preliminary_data_path, finals_data_path)
    # train_loader, test_loader = load_data(merged_data_path, test_size=0.1)
    # checkpoint_model = torch.load("./best_model_3.pth")
    # _, best_model = train_model(train_loader, test_loader, initial_model=checkpoint_model, learning_rate=5e-4, tolerance=20)

    # compose the model list for ensemble learning
    model_list = []
    best_model = torch.load("./best_model_2.pth")  # balance one
    model_list.append(best_model)
    best_model = torch.load("./best_model_3.pth")  # less false alarm
    model_list.append(best_model)
    best_model = torch.load("./best_model_4.pth")  # sensitive one (more false alarm but)
    model_list.append(best_model)
    # assign weight to the models
    weight_list = [0.7, 0.2, 0.1]  # balance one, less false alarm, sensitive one, sum up to 1.0

    # Test the model
    # data_path = '../../data/event_2/Train_Finals_tracks_graph.npy'
    # _, test_loader = load_data(data_path, test_size=1.0)
    # test_model(model_list, test_loader)

    # output predictions
    data_path = '../../data/event_2/Test_Finals.csv'
    npy_path = extract_event_2_data_from_csv(data_path)
    _, test_loader = load_data(npy_path, test_size=1.0)
    predictions = predict(model_list, test_loader)
    append_predictions(data_path, predictions)

    # best_model = torch.load("./best_model_1.pth")
    # test_model(best_model, test_loader)
