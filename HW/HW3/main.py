import numpy as np

from Settings import Setting
from models import resnet,FocalLoss,resnet18
from  utils import get_transforms,set_rand_seed,compute_gradient_norm
from DataSet import FoodDataSet
from tqdm.auto import tqdm
import os
import torch.nn as nn

import logging
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_exp_name = "resnet"
def get_file_paths(folder_name):
    train_folder = os.path.join(folder_name, 'training')
    validation_folder = os.path.join(folder_name, 'validation')

    train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]
    validation_files = [os.path.join(validation_folder, f) for f in os.listdir(validation_folder) if os.path.isfile(os.path.join(validation_folder, f))]
    train_files.extend(validation_files)
    return train_files
def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def kf_train( n_epochs, patience,dataset,n_splits=5):
    kf = KFold(n_splits=n_splits,shuffle=True)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        logging.info(f"Fold {fold + 1}/{n_splits}")
        train_data, val_data = dataset[train_index], dataset[val_index]
        train_tfm, test_tfm = get_transforms()
        train_data_set = FoodDataSet(files=train_data, tfm=train_tfm)
        val_data_set = FoodDataSet(files=val_data, tfm=test_tfm)
        train_loader = DataLoader(train_data_set, batch_size=Set.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        valid_loader = DataLoader(val_data_set, batch_size=Set.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        model = resnet18()
        # Re-initialize the model and optimizer for each fold
        model.apply(reset_weights)
        alpha=torch.tensor([1.0,1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
        alpha=nn.functional.softmax(alpha,0)
        criterion = FocalLoss(alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=Set.lr, weight_decay=Set.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=3e-7,T_max=175)
        train(model, n_epochs, patience, criterion, optimizer, train_loader, valid_loader, fold=fold,scheduler=scheduler)
def train(model, n_epochs, patience, criterion, optimizer, train_loader, valid_loader, fold=0,scheduler=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stale = 0
    best_acc = 0
    writer = SummaryWriter()

    for epoch in range(n_epochs):
        model.train()
        model.to(device)
        train_loss, train_accs = [], []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc.item())

        if scheduler:
            scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_norm = compute_gradient_norm(model)
        writer.add_scalar(f"Norm/Train", train_norm, epoch)
        writer.add_scalar(f'Loss/train', train_loss, epoch)
        writer.add_scalar(f'Acc/train', train_acc, epoch)

        logging.info(f"[Train|{epoch + 1}/{n_epochs}] loss={train_loss:.5f}, acc={train_acc:.5f}")

        # Validation
        model.eval()
        valid_loss, valid_accs = [], []

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                imgs, labels = batch
                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc.item())

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        writer.add_scalar(f'Loss/valid', valid_loss, epoch)
        writer.add_scalar(f'Acc/valid', valid_acc, epoch)

        logging.info(f"[ Valid | {epoch + 1}/{n_epochs} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # Save the best model and check for early stopping
        if valid_acc > best_acc:
            best_acc = valid_acc
            stale = 0
            torch.save(model.state_dict(), os.path.join(_exp_name,f"best_model_fold{fold + 1}.ckpt"))  # 根据折叠命名
            logging.info(f"Best model found at epoch {epoch + 1}, saving model")
        else:
            stale += 1
            if stale > patience:
                logging.info(f"No improvement for {patience} epochs, early stopping")
                break


if __name__=="__main__":
    set_rand_seed(42)
    Set=Setting()
    # train_tfm,test_tfm,=get_transforms()
    # train_set=FoodDataSet(file_path=os.path.join(Set.data_dir,"training"),tfm=train_tfm)
    # train_loader=DataLoader(train_set,batch_size=Set.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    # valid_set=FoodDataSet(file_path=os.path.join(Set.data_dir,"validation"),tfm=test_tfm)
    # valid_loader=DataLoader(valid_set,batch_size=Set.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    dataset=get_file_paths("food11")
    dataset=np.array(dataset)
    kf_train(Set.n_epoch,Set.patience,dataset,5)
    # train(model,Set.n_epoch,Set.patience,criterion,optimizer,train_loader,valid_loader,scheduler=scheduler)


