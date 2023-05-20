import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import torchvision.models as models

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # Convolutional Neural Network Layer 
        self.convolutaional_neural_network_layers = nn.Sequential(
                # Here we are defining our 2D convolutional layers
                # We can calculate the output size of each convolutional layer using the following formular
                # outputOfEachConvLayer = [(in_channel + 2*padding - kernel_size) / stride] + 1
                # We have in_channels=1 because our input is a grayscale image
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1), # (N, 1, 28, 28) 
                nn.ReLU(),
                # After the first convolutional layer the output of this layer is:
                # [(28 + 2*1 - 3)/1] + 1 = 28. 
                nn.MaxPool2d(kernel_size=2), 
                # Since we applied maxpooling with kernel_size=2 we have to divide by 2, so we get
                # 28 / 2 = 14
          
                # output of our second conv layer
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                # After the second convolutional layer the output of this layer is:
                # [(14 + 2*1 - 3)/1] + 1 = 14. 
                nn.MaxPool2d(kernel_size=2) 
                # Since we applied maxpooling with kernel_size=2 we have to divide by 2, so we get
                # 14 / 2 = 7
        )

        # Linear layer
        self.linear_layers = nn.Sequential(
                # We have the output_channel=24 of our second conv layer, and 7*7 is derived by the formular 
                # which is the output of each convolutional layer
                nn.Linear(in_features=24*7*7, out_features=64),          
                nn.ReLU(),
                #nn.Dropout(p=0.2), # Dropout with probability of 0.2 to avoid overfitting
                nn.Linear(in_features=64, out_features=10) # The output is 10 which should match the size of our class
        )

    # Defining the forward pass 
    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        # After we get the output of our convolutional layer we must flatten it or rearrange the output into a vector
        x = x.view(x.size(0), -1)
        # Then pass it through the linear layer
        x = self.linear_layers(x)
        return x

model = Network()
model = torch.load("/Users/khavishgovind/Documents/EEE4114/ML Project/eee4114f-project/Khavish/model.pt")
#transforms.Normalize((0.1307,),(0.3081,))

transform_1 = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale()])
test_set = datasets.ImageFolder('/Users/khavishgovind/Documents/EEE4114/ML Project/eee4114f-project/Khavish/testData/',transform=transform_1)
testLoader = torch.utils.data.DataLoader(test_set, batch_size=80, shuffle=True,) #transforms.Normalize((0.1307,),(0.3081,)

optimizer = optim.Adam(model.parameters())#optim.SGD(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()#nn.MSELoss()

test_loss,accuracy_total_test = [],[]
total_test_loss = 0
match = []

# Testing our model
model.eval()
total = 0
for idx, (images, labels) in enumerate(testLoader):
    preds = model(images)
    loss = criterion(preds, labels)
    total_test_loss += loss.item()

    preds = torch.nn.functional.softmax(preds, dim=1)

    for i, p in enumerate(preds):
        #print("--------")
        #fig = plt.figure()
        #plt.imshow(images[i][0])
        #plt.title("True Label: {}".format(labels[i]))
        #plt.xlabel("Prediction Label: {}".format(torch.max(p.data, 0)[1]))
        #plt.show()
        match.append(torch.max(p.data, 0)[1])
        if labels[i] == torch.max(p.data, 0)[1]:
            total = total + 1

accuracy_test = total / len(test_set)
accuracy_total_test.append(accuracy_test)

total_test_loss = total_test_loss / (idx + 1)
test_loss.append(total_test_loss)


print("------------------")
print("------------------")

print("Test loss: {:.4f}  ".format(total_test_loss),"Test accuracy: {:.4f}  ".format(accuracy_test))
cm = confusion_matrix(labels.detach().numpy(), np.array(match))
ConfusionMatrixDisplay(cm).plot()

cf_matrix = confusion_matrix(labels.detach().numpy(), np.array(match))
class_names = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

# Create pandas dataframe
dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))

# Create heatmap
#sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")

plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), 
plt.xlabel("Predicted Class")
plt.show()