import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


t = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])

# Load datasets for training and testing.
testData = datasets.ImageFolder('testData/', transform=t)

# Load model
model = torch.load("modelNN")

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

# The tensor images has the shape [10000, 1, 28, 28]. Reshape the tensor to
# [10000, 784] as our model expected a flat vector.
data = images.view(n, -1)

# Use our model to compute the class scores for all images. The result is a
# tensor with shape [10000, 10]. Row i stores the scores for image images[i].
# Column j stores the score for class j.
predictions = model(data)

# For each row determine the column index with the maximum score. This is the
# predicted class.
predicted_classes = torch.argmax(predictions, dim=1)

# Accuracy = number of correctly classified images divided by the total number
# of classified images.
accuracy = sum(predicted_classes.numpy() == labels.numpy()) / n

print("The accuracy of the model is: "+str(accuracy))