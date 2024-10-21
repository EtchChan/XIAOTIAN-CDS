import torch
from pytorch_model_summary import summary
from sympy.core.random import sample

# from src.NN.transformer_classifier import MultiHeadAttentionClassifier  # Import customized model class
from src.GNN.GAT_classifier import GAT, DroneRadarDataset  # Import customized model class

# Load your model
# model = MultiHeadAttentionClassifier(6, 3, 2)
model = GAT(6, 16, 2)
state_dict = torch.load('../GNN/best_model_1.pth').state_dict()
model.load_state_dict(state_dict)

# Print model summary
# generate an graph input for the GAT model
dataset = DroneRadarDataset("../../data/event_2/raw_tracks_graph.npy")
sample_graph = dataset[0]
model_summary = summary(model, sample_graph)

# model_summary = summary(model, torch.zeros((1, 15, 6)))
print(model_summary)