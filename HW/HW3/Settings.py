import torch
class Setting:
    def __init__(self,n_epoch=200,patience=30,lr=3e-4,weight_decay=1e-5,batch_size=32,data_dir="food11",num_classes=11):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.n_epoch=n_epoch
        self.patience=patience
        self.lr=lr
        self.weight_decay=weight_decay
        self.batch_size=batch_size
        self.data_dir=data_dir
        self.num_classes=num_classes