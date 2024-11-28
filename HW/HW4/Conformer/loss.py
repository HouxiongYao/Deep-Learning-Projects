import torch
import torch.nn as nn
import torch.nn.functional as F
class AdMSoftmaxLoss(nn.Module):
    def __init__(self,input_dim,num_of_classes,s=30,m=0.5):
        '''
        :param input_dim: dim of the input
        :param num_of_classes: classes number
        :param s: scale factor
        :param m: margin
        '''
        super().__init__()
        self.s=s
        self.m=m
        self.fc=nn.Linear(input_dim,num_of_classes,bias=False)

    def forward(self,x,labels=None):
        '''
        :param x:  inputs,(sample_num,feature_dim)
        :return: loss,logits if label is not None, otherwise logits
        '''
        for w in self.fc.parameters():
            w=F.normalize(w,dim=1)
        x=F.normalize(x,dim=1)
        total_cos_value=self.fc(x)#(sample_num,num_of_classes)
        if labels is None:
            return total_cos_value
        numerator=self.s*(torch.gather(total_cos_value,dim=1,index=labels.view(len(labels),-1))-self.m)
        indices=torch.arange(0,total_cos_value.size(1)).to(labels.device)
        mask=indices!=(labels).unsqueeze(1)
        result=total_cos_value.masked_select(mask).reshape(total_cos_value.size(0),total_cos_value.size(1)-1)
        denominator=torch.exp(numerator)+torch.sum(torch.exp(self.s*result),dim=1)
        loss=numerator-torch.log(denominator)
        return -torch.mean(loss),total_cos_value