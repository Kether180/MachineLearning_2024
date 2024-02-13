import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


def show_samples(X, y, num, prediction=None, sort=True, cols=32, width_mul=1):
    if prediction is None:
        height_mul = 1
    else:
        height_mul = 2
        
    if sort: 
        idx = np.argsort(y[:num])
        X = X[idx]
        if prediction is not None:
            prediction = prediction[idx]
    fig, ax = plt.subplots(nrows=num//cols, ncols=cols, figsize=(width_mul*cols, height_mul*num//cols))
    for i in range(num):
        ax[i//cols, i%cols].axis('off')
        ax[i//cols, i%cols].imshow(X[i].reshape((28, 28)), cmap='gray')
        if prediction is not None:
            classes = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
            ax[i//cols, i%cols].set_title(f'{classes[prediction[i]]}/{classes[y[i]]}')

class Linear(nn.Module):
    """Define an affine transformation (linear and bias) neural network without an activation function. 

    """
    def __init__(self):
        super().__init__()
        #Define archtechture
        # matrix of linear activation 784 inputs to 10 outputs
        self.layer_1 = nn.Linear(784, 10)

    def forward(self, x):
        # Flatten input to vector
        x = torch.flatten(x, start_dim=1)
        # Make prediction
        x = self.layer_1(x)
        return x


class MLPBasic(nn.Module):
    def __init__(self):
        super().__init__()
         #Define a two layer network archtechture
        self.layer_1 = nn.Linear(784, 100)
        self.layer_2 = nn.Linear(100, 10)

    def forward(self, x):
         #Prediction
         
        #input to vector
        x = torch.flatten(x, start_dim=1)

        #Prediction through first layer and relu activation
        x = self.layer_1(x)
        x = F.relu(x)


        #Prediction through second layer with relu activation
        x = self.layer_2(x)

        return F.relu(x)


class CNNBasic(nn.Module):
    """Define a CNN with two covolutional layers and two linear layers
    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        #Define network archtechture with two convolutional layers and two linear layers
       
        # 8 kernels each with size 9
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=9) 
       
        # layer that for each of the  8 channels from previous layer
        #  having 16 kernels each of size 3
        self.conv_3 = nn.Conv2d(8, 16, kernel_size=3) 
        self.linear_1 = nn.Linear(4 ** 2 * 16, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        #Prediction
        x = F.relu(self.conv_1(x)) #relu activation on the first layer
        x = F.max_pool2d(x, kernel_size=2, stride=2) 

        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #make input into one vector
        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class CNN4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.layer_1 = nn.Linear(in_features=16 * 32, out_features=80)
        self.layer_2 = nn.Linear(in_features=80, out_features=10)

    def forward(self, picture):
        imageConv1 = F.relu(
            self.conv_1(picture)
        )  # 1 image is given, 8 output channels , kernel size 9, size of the new images is 28-9+1 = 20
        imageConv2 = F.relu(
            self.conv_2(imageConv1)
        )  # 8 images is given, 16 channels out, kernel size 3, new image size = 10-3+1 = 8 divides the size of the image with 2
        maxPool2 = F.max_pool2d(
            imageConv2, 2, 2
        )  # divides the imagesize by 2, image size = 4
        imageConv3 = F.relu(
            self.conv_3(maxPool2)
        )  # 8 images is given, 16 channels, kernel size 3, new image size = 10-3+1 = 8 divides the size of the image with 2
        imageConv3 = F.relu(
            self.conv_4(imageConv3)
        )  # 8 images is given, 16 comes out, kernel size 3, new image size = 10-3+1 = 8 divides the size of the image with 2
        maxPool4 = F.max_pool2d(imageConv3, 2, 2)
        imageFlatten = torch.flatten(maxPool4, start_dim=1)
        linearImage1 = F.relu(self.layer_1(imageFlatten))
        return self.layer_2(linearImage1)



class TopCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=9, padding=4)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn_2 = nn.BatchNorm2d(32)

        self.drop_1 = nn.Dropout2d(p=0.2)

        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn_4 = nn.BatchNorm2d(64)

        self.drop_2 = nn.Dropout2d(p=0.2)

        self.linear_1 = nn.Linear(5 ** 2 * 64, 100)

        self.drop_3 = nn.Dropout(p=0.2)

        self.linear_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.bn_1(x)
        x = F.relu(self.conv_2(x))
        x = self.bn_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.drop_1(x)

        x = F.relu(self.conv_3(x))
        x = self.bn_3(x)
        x = F.relu(self.conv_4(x))
        x = self.bn_4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.drop_2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.drop_3(x)

        x = self.linear_2(x)

        return x

