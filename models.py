import torch
import torch.nn as nn
import timm

# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#        # print(f"MLP input shape before flattening: {x.shape}")  # Debug print
#         x = x.view(x.size(0), -1)  # Flatten input
#         #print(f"MLP input shape after flattening: {x.shape}")  # Debug print
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
class MLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):  #for MNIST
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_model(model_name, num_classes):
    if model_name == "resnet18":
        return timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    elif model_name == "mlp":
        #return MLP(input_size=3072, hidden_size=100,num_classes=num_classes)
        return MLP(input_size=784, num_classes=num_classes)
    elif model_name == "lenet":
        return LeNet(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
