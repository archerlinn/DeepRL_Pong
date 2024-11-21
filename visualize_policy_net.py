import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

# Define PolicyNetwork
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=3):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_probs = F.softmax(x, dim=1)
        return action_probs

# Function to visualize individual neurons
def visualize_neurons(layer, title_prefix="Neuron Visualization"):
    """
    Visualize weights of individual neurons in a layer.
    
    Args:
        layer (torch.nn.Linear): The layer to visualize.
        title_prefix (str): Prefix for the plot titles.
    """
    weights = layer.weight.detach().cpu().numpy()  # Extract weights
    num_neurons = weights.shape[0]  # Number of output neurons in the layer

    for i in range(num_neurons):
        plt.figure(figsize=(8, 6))
        plt.plot(weights[i], marker="o")
        plt.title(f"{title_prefix} - Neuron {i+1}")
        plt.xlabel("Input Index")
        plt.ylabel("Weight Value")
        plt.grid()
        plt.show()

# Load the saved policy network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = PolicyNetwork().to(device)

try:
    policy_net.load_state_dict(torch.load("policy_net_saved_episode.pth", map_location=device))
    print("Loaded pre-trained model successfully.")
except FileNotFoundError:
    print("No saved model found. Please train and save the model first.")
    exit()

# Visualize individual neurons in the first layer
visualize_neurons(policy_net.fc1, "First Layer Neuron Weights")

# Visualize individual neurons in the second layer
visualize_neurons(policy_net.fc2, "Second Layer Neuron Weights")
