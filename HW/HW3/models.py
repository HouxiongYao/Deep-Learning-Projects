import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Union
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc=nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
    def forward(self,x):
        out=self.cnn(x)
        out=out.view(out.size()[0],-1)
        return self.fc(out)

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else: self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
    def forward(self,x):
        Y=F.relu(self.bn1(self.conv1(x)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            x=self.conv3(x)
        Y+=x
        return F.relu(Y)
def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 softmax 概率
        p = torch.softmax(inputs, dim=1)
        # 获取目标类别的概率
        p_t = p.gather(1, targets.view(-1, 1)).view(-1)
        try:
            loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-12)
        except:
            at=self.alpha.to(inputs.device).gather(0,targets)
            loss = -at * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-12)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def resnet(in_channels=3,classes=11):
    b1=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
                     nn.BatchNorm2d(64),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                     )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # fc=nn.Sequential(nn.Dropout(0.5),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,classes))
    fc=nn.Linear(128,classes)
    net=nn.Sequential(b1,b2,b3,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),fc)
    return net
def resnet18(in_channels=3,classes=11):
    b1=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
                     nn.BatchNorm2d(64),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                     )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    fc=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,classes))
    net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),fc)
    return net
if __name__=="__main__":
    logits = torch.tensor([[1.0, 0.5, 0.2], [0.1, 2.0, 0.1],[0.2,0.3,0.4]], requires_grad=True)  # 模型输出
    targets = torch.tensor([0, 1,2])  # 真实标签，表示类别 0 和 1
    alpha=torch.tensor([0.25,0.5,0.75])
    # 创建 Focal Loss 实例
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0, reduce=True, logits=True)

    # 计算损失
    loss = focal_loss(logits, targets)
    print(loss.item())
