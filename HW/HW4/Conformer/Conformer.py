import torch
import torch.nn as nn
from .attention import RelPartialLearnableMultiHeadAttn
from .activation import Swish,GLU
from .convolution import DepthwiseConv1d,PointwiseConv1d
from .pooling import SelfAttentionPooling
from .loss import AdMSoftmaxLoss
device="cuda" if torch.cuda.is_available() else "cpu"


class atten_module(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        # 初始化 rw 和 rr 作为模型的参数
        self.rw = nn.Parameter(torch.randn(n_head, d_head))
        self.rr = nn.Parameter(torch.randn(n_head, d_head))
        self.layer_norm = nn.LayerNorm(d_model)
        self.atten = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout)
        self.dropout = nn.Dropout(dropout)
        # 在初始化时定义 r 作为模型参数
        self.r = nn.Parameter(torch.randn(1, self.d_model))  # 初始化为 (1, d_model)

    def forward(self, x):
        x = self.layer_norm(x)
        # 根据 x 的第一个维度调整 r 的形状
        r_len = x.size(0)  # 获取 x 的 seq_len (第一个维度)
        self.r.expand(r_len, self.d_model)  # 扩展 r 的形状

        # 进行注意力计算
        x = self.atten(x, self.r.expand(r_len, self.d_model), self.rw, self.rr)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, n_head,d_model, d_head,dropout=0.3):
        super().__init__()
        self.conv_module=nn.Sequential(
            nn.LayerNorm(d_model),
            PointwiseConv1d(d_model,2*d_model),
            GLU(2*d_model),
            DepthwiseConv1d(2*d_model,2*d_model,kernel_size=3),
            nn.BatchNorm1d(2*d_model),
            Swish(),
            PointwiseConv1d(2*d_model,d_model),
            nn.Dropout(dropout),
        )
        self.atten_module = atten_module(n_head,d_model,d_head)
        self.atten_module.to(device)
        self.feedforward_module=nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,4*d_model),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model,d_model),
            nn.Dropout(dropout),
        )
    def forward(self,x):#(bsz,l,d_model)
        x=x+0.5*self.feedforward_module(x)#(bsz,l,d_model)
        x=x.permute(1,0,2)
        x=x+self.atten_module(x)#(length,bsz,d_model)
        x=x.permute(1,0,2)
        x=x+0.5*self.feedforward_module(x)
        x=torch.layer_norm(x,[x.size(-1)])#(bsz,l,d_model)
        return x
class Comformer_net(nn.Module):
    def __init__(self,n_head, d_model, d_head, dropout=0.1,n_blocks=2,n_spks=600):
        super().__init__()
        self.d_model=d_model
        conformer_block=ConformerBlock(n_head,d_model,d_head)
        self.subsampling=nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2,stride=2),#(bsz,channels,length)
        )
        self.prenet=nn.Sequential(
            nn.Linear(40,d_model),
            nn.Dropout(dropout),
        )
        self.conformer=nn.Sequential(
            *[conformer_block for _ in range(n_blocks)]
        )
        self.loss=AdMSoftmaxLoss(d_model,n_spks,s=30,m=0.5)
        self.pooling=SelfAttentionPooling(d_model)
    def forward(self,mels,labels=None):#(bsz,l,d_model)
        mels=mels.permute(0,2,1)#(bsz,d_model,l)
        mels=self.subsampling(mels).permute(0,2,1)#(bsz,l,d_model)
        mels=self.prenet(mels)#(basz,l,d_model)
        mels=self.conformer(mels)#（basz,l,d_model）
        outputs=self.pooling(mels)
        if labels is not None:
            loss,logits=self.loss(outputs,labels)
            return loss,logits
        else:
            return self.loss(outputs)


if __name__=="__main__":
    # 设置模型参数
    n_head = 4
    d_model = 64
    d_head = 16
    dropout = 0.1
    n_blocks = 2
    n_spks = 600

    # 初始化模型
    model = Comformer_net(n_head=n_head, d_model=d_model, d_head=d_head, dropout=dropout, n_blocks=n_blocks,
                          n_spks=n_spks)

    # 生成随机输入数据：假设输入是 (batch_size, length, feature_dim) 的三维张量
    batch_size = 8
    length = 100  # 输入序列长度
    feature_dim = 40  # 输入特征维度
    mels = torch.randn(batch_size, length, feature_dim)

    # 将输入数据传入模型
    output = model(mels)

    # 输出结果
    print("输出形状:", output.shape)
    print("输出内容:", output)

