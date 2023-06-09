import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

t = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])

# Load datasets for training and testing.
testData = datasets.ImageFolder('testData/', transform=t)

class Net(torch.nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(1, 4, kernel_size=7, stride=1, padding=7//2),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.MaxPool2d(2),
            # Defining another 2D convolution layer
            torch.nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=7//2),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2), 
            torch.nn.MaxPool2d(2),  
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(196, 125),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.Linear(125, 80),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.Linear(80, 10),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
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
#
#fig = plt.figure()
#plt.imshow(images[0][0], cmap='gray')
#plt.title("Test Data: {}".format(labels[0]))
#plt.yticks([])
#plt.xticks([])
#plt.show()
#

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

cm = confusion_matrix(labels.detach().numpy(), np.array(predicted_classes))
ConfusionMatrixDisplay(cm).plot()

cf_matrix = confusion_matrix(labels.detach().numpy(), np.array(predicted_classes))
class_names = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

# Create pandas dataframe
dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))

# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")

plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), 
plt.xlabel("Predicted Class")
plt.show()