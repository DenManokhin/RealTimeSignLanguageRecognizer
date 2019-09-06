import torch
import torch.utils
import torch.nn as nn
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TYolo(nn.Module):
    def __init__(self):
        super(TYolo,self).__init__()
        
        self.conv1=nn.Conv2d(3,16,3,padding=0)
        self.conv2=nn.Conv2d(16,32,3,padding=0)
        self.conv3=nn.Conv2d(32,64,3,padding=0)
        self.conv4=nn.Conv2d(64,128,3,padding=0)
        self.conv5=nn.Conv2d(128,256,3,padding=0)
        self.conv6=nn.Conv2d(256,512,3,padding=0)
        self.conv7=nn.Conv2d(512,1024,3,padding=0)
        self.conv8=nn.Conv2d(1024,32,3,padding=0)
        self.conv9=nn.Conv2d(32,30,1,padding=0)
        self.pool1=nn.MaxPool2d(2,2)
        self.pool2=nn.MaxPool2d((2,2),(1,1))
        self.fc1=nn.Linear(5070,4096)
        self.fc2=nn.Linear(1024,256)
        self.fc3=nn.Linear(256,64)
        self.fc4=nn.Linear(64,16)
        self.fc5=nn.Linear(16,4)
        self.fc6=nn.Linear(4096,1024)

    def forward(self, x):
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv1(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv2(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv3(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv4(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv5(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=F.leaky_relu(self.conv6(x))
        x=F.pad(x,(0,1,0,1),'reflect')
        x=self.pool2(x)
        x=F.pad(x,(1,1,1,1),'reflect')
        x=F.leaky_relu(self.conv7(x))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=F.leaky_relu(self.conv8(x))
        x=self.conv9(x)
        x=predict_transform(x)
        x=x.view(1,5070)
        x=self.fc1(x)
        x=self.fc6(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        x=self.fc5(x)
        return x
    
