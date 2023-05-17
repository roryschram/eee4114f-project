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

# Hyperparameters for our network
input_size = 784
hidden_sizes = [700, 600, 500, 400, 300, 200, 100]
output_size = 10
# Build a feed-forward network
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_sizes[0]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[3], hidden_sizes[4]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[4], hidden_sizes[5]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[5], hidden_sizes[6]),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_sizes[6], output_size),
)   
print(model)


# Use Adam as optimizer.
opt = torch.optim.Adam(params=model.parameters(), lr=0.01)

# Use mean squared error for as loss function.
loss_fn = torch.nn.CrossEntropyLoss()

# We train the model with batches of 500 examples.
batch_size = 500
train_loader = torch.utils.data.DataLoader(mnist_training, batch_size=batch_size, shuffle=True)

losses = []

for epoch in range(10):
    for imgs, labels in train_loader:
        n = len(imgs)
        # Reshape data from [500, 1, 28, 28] to [500, 784] and use the model to make predictions.
        predictions = model(imgs.view(n,-1))  
        # Compute the loss.
        loss = loss_fn(predictions, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print(f"Epoch: {epoch}, Loss: {float(loss)}")

# Plot learning curve.
#plt.plot(losses)
#plt.show()



# Determine the accuracy of our clasifier
# =======================================

# Load all 10000 images from the validation set.
n = 10000
loader = torch.utils.data.DataLoader(mnist_val, batch_size=n)
dataiter = iter(loader)
images, labels = next(dataiter)

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

print("The accuracy of the model is: "+str(accuracy)+"%")

torch.save(model, "modelNN")