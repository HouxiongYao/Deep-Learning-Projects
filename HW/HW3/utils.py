import torch
import numpy as np
import torchvision.transforms.v2 as transforms
from torch.utils.data import default_collate

def set_rand_seed(myseed=6666):
    # 设置 PyTorch 的 CuDNN 以确保确定性行为
    torch.backends.cudnn.deterministic = True
    # 禁用 CuDNN 的基准测试，防止每次运行时选择不同的算法
    torch.backends.cudnn.benchmark = False

    # 设置 NumPy 的随机种子
    np.random.seed(myseed)
    # 设置 PyTorch 的随机种子（CPU）
    torch.manual_seed(myseed)

    # 如果 CUDA 可用，设置所有 GPU 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
def get_transforms():
    # Normally, We don't need augmentations in testing and validation.
    # All we need here is to resize the PIL image and transform it into Tensor.
    test_tfm = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((224,224)),
        # transforms.ColorJitter(contrast=1.5),
        transforms.ToDtype(torch.float32,scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # However, it is also possible to use augmentation in the testing phase.
    # You may use train_tfm to produce a variety of images and then test using ensemble methods
    train_tfm = transforms.Compose([
        transforms.PILToTensor(),
        transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm,test_tfm

def Preprocess(batch, num_classes):
    cutmix = transforms.CutMix(num_classes=num_classes)
    mixup = transforms.MixUp(num_classes=num_classes)
    cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])

    # 默认合并批次
    images, labels = default_collate(batch)  # 这里假设返回值是图像和标签的元组

    # 应用选择的增强方法
    return cutmix_or_mixup(images, labels)
def compute_gradient_norm(model):
    if any(param.grad is None for param in model.parameters()):
        raise ValueError
    gradients=[param.grad for param in model.parameters() if param.grad is not None]
    gradients_norm=torch.sqrt(sum(torch.sum(g**2) for g in gradients)).item()
    return gradients_norm


