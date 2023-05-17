import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

#import helper
#matplotlib inline
#config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np

# moves your model to train on your gpu if available else it uses your cpu
#device = ("cuda" if torch.cuda.is_available() else "cpu")

# Define transform to normalize data
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
train_set = datasets.MNIST("./", download=True, train=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(train_set, batch_size=600, shuffle=True)

validation_set = datasets.MNIST("./", download=True, train=False, transform=transform)
validationLoader = torch.utils.data.DataLoader(validation_set, batch_size=600, shuffle=True)


training_data = enumerate(trainLoader)
batch_idx, (images, labels) = next(training_data)
#print(type(images)) # Checking the datatype 
#print(images.shape) # the size of the image
#print(labels.shape) # the size of the labels

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
                # We have the output_channel=24 of our second conv layer, and 7*7 is derived by the formula 
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

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()#nn.MSELoss()

epochs = 10
train_loss, val_loss = [], []
accuracy_total_train, accuracy_total_val = [], []

for epoch in range(epochs):
   
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    
    total = 0
    # training our model
    for idx, (image, label) in enumerate(trainLoader):

        optimizer.zero_grad()

        n = len(image)
        pred = model(image)
        oneHotLabels = F.one_hot(label, num_classes=10)
        #loss = criterion(pred, oneHotLabels.to(torch.float32))
        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
                
    accuracy_train = total / len(train_set)
    accuracy_total_train.append(accuracy_train)

    total_train_loss = total_train_loss / (idx + 1)
    train_loss.append(total_train_loss)
    
    # validating our model
    model.eval()
    total = 0
    for idx, (image, label) in enumerate(validationLoader):
        pred = model(image)
        #loss = criterion(pred, oneHotLabels.to(torch.float32))
        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy_val = total / len(validation_set)
    accuracy_total_val.append(accuracy_val)

    total_val_loss = total_val_loss / (idx + 1)
    val_loss.append(total_val_loss)

    #if epoch % 5 == 0:
    print("Epoch: {}/{}  ".format(epoch, epochs),
            "Training loss: {:.4f}  ".format(total_train_loss),
            "Validation loss: {:.4f}  ".format(total_val_loss),
            "Train accuracy: {:.4f}  ".format(accuracy_train),
            "Validation accuracy: {:.4f}  ".format(accuracy_val))
print("-------------------------------------------------")    

transform_1 = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])
test_set = datasets.ImageFolder('testData/',transform=transform_1)
testLoader = torch.utils.data.DataLoader(test_set, batch_size=80, shuffle=True)

#fig = plt.figure()
#for i in range(9):
#    plt.subplot(3,3,i+1)
#    plt.tight_layout()
#    plt.imshow(images[i][0], cmap='inferno')
#    plt.title("Test Data: {}".format(labels[i]))
#    plt.yticks([])
#    plt.xticks([])
#    plt.show()

test_loss,accuracy_total_test = [],[]
total_test_loss = 0

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
        #plt.imshow(images[i][0], cmap='inferno')
        #plt.title("Test Data: {}".format(labels[i]))
        #print(torch.max(p.data, 0)[1])
        #plt.xlabel("Prediction Label: {}".format(torch.max(p.data, 0)[1]))
        #plt.show()
        #print("--------")
        #print(torch.max(p.data, 0)[1])
        #print("Asscoaited labels is")
        #print(labels[i])
        #print("-------")
        if labels[i] == torch.max(p.data, 0)[1]:
            total = total + 1

accuracy_test = total / len(test_set)
accuracy_total_test.append(accuracy_test)

total_test_loss = total_test_loss / (idx + 1)
test_loss.append(total_test_loss)

print("Test loss: {:.4f}  ".format(total_test_loss),"Test accuracy: {:.4f}  ".format(accuracy_test))