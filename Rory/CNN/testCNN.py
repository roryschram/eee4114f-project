import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

t = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])

# Load datasets for training and testing.
testData = datasets.ImageFolder('testData/', transform=t)

class Net(torch.nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Defining another 2D convolution layer
            torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(4 * 7 * 7, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# Load model
model = Net()
model.load_state_dict(torch.load("modelCNN"))

# Determine the accuracy of our clasifier
# =======================================

# Load all 10000 images from the validation set.
n = 80
loader = torch.utils.data.DataLoader(testData, batch_size=n, shuffle=True)
dataiter = iter(loader)
images, labels = next(dataiter)

fig = plt.figure()
plt.imshow(images[0][0], cmap='gray')
plt.title("Test Data: {}".format(labels[0]))
plt.yticks([])
plt.xticks([])
plt.show()


# Use our model to compute the class scores for all images. The result is a
# tensor with shape [10000, 10]. Row i stores the scores for image images[i].
# Column j stores the score for class j.
predictions = model(images)

# For each row determine the column index with the maximum score. This is the
# predicted class.
predicted_classes = torch.argmax(predictions, dim=1)

# Accuracy = number of correctly classified images divided by the total number
# of classified images.
accuracy = sum(predicted_classes.numpy() == labels.numpy()) / n

print("The accuracy of the model is: "+str(accuracy))