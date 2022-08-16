import os
# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Pytorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 1

model_file = 'LeNet5_model.pth'

prof_activity = {
    'cpu'   : ProfilerActivity.CPU,
    'cuda'  : ProfilerActivity.CUDA
}

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

#Loading the dataset and preprocessing

root_directory='./MNIST'
train_dataset = torchvision.datasets.MNIST(root = root_directory,
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = True)


test_dataset = torchvision.datasets.MNIST(root = root_directory,
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                          download=True)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

#Defining the convolutional neural network
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        #super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def train(model,device=torch.device('cpu')):
    #Setting the loss function
    cost = nn.CrossEntropyLoss()

    #Setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #this is defined to print how many steps are remaining when training
    total_step = len(train_loader)

    prof_sort_by = [ f"self_{device.type}_time_total", f"{device.type}_time_total" ]

    # Train the model
    with profile(activities=[prof_activity[device.type]], record_shapes=True) as prof:
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    #Forward pass
                    with record_function("model_train_forward"):
                        outputs = model(images)
                    with record_function("model_train_loss_func"):
                        loss = cost(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #if (i+1) % 400 == 0:
                    #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    #           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    for _sort_by in prof_sort_by:
        print(prof.key_averages().table(sort_by=_sort_by, row_limit=50))
    return model


model = ConvNeuralNet(num_classes).to(device)
if False: #os.path.exists(model_file):
    print(f"Loading model parameters from {model_file}")
    model.load_state_dict(torch.load(model_file))
else:
    print(f"Training model....")
    # Define model and train it
    model = train(model,device)
    print(f"Saving model to {model_file}")
    torch.save(model.state_dict(), model_file)

print(f"Training finished!")

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
#with torch.no_grad():
#    correct = 0
#    total = 0
#    for images, labels in test_loader:
#        images = images.to(device)
#        labels = labels.to(device)
#        outputs = model(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()

 #   print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
