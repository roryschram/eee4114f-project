import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.ToTensor()

# Load datasets for training and testing.
mnist_training = datasets.MNIST("./", train=True, download=True, transform=t)
mnist_val = datasets.MNIST("./", train=False, download=True, transform=t)

trainLoader = torch.utils.data.DataLoader(mnist_training, batch_size=600, shuffle=True)
validationLoader = torch.utils.data.DataLoader(mnist_val, batch_size=600, shuffle=True)


class Net(torch.nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(1, 8, (5, 5)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(8, 8, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 3)),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(324, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# defining the model
model = Net()

# Use Adam as optimizer.
opt = torch.optim.Adam(params=model.parameters(), lr=0.01)

# Use mean squared error for as loss function.
loss_fn = torch.nn.CrossEntropyLoss()

print(model)


# We train the model with batches of 500 examples.
batch_size = 1000
train_loader = torch.utils.data.DataLoader(mnist_training, batch_size=batch_size, shuffle=True)
losses = []

for epoch in range(10):
    for imgs, labels in train_loader:
        n = len(imgs)
        # Reshape data from [500, 1, 28, 28] to [500, 784] and use the model to make predictions.
        predictions = model(imgs)  
        # Compute the loss.
        loss = loss_fn(predictions, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print(f"Epoch: {epoch}, Loss: {float(loss)}")

torch.save(model.state_dict(), "modelCNN")