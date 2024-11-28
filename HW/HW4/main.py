from torch.nn.functional import dropout
from Conformer import Comformer_net
from Config import Config
from tqdm.auto import tqdm
import os
import logging
from conformer1 import Conformer
import torch
from tqdm import tqdm
from utlis import get_cosine_schedule_with_warmup, get_loader, set_seed
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_exp_name = "models"
def train(model, n_epochs, patience,optimizer, train_loader, valid_loader, fold=0,scheduler=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stale = 0
    best_acc = 0
    writer = SummaryWriter()
    print(device)
    for epoch in range(n_epochs):
        model.train()
        model.to(device)
        train_loss, train_accs = [], []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            # loss,logits= model(imgs.to(device),imgs.size(1),labels.to(device))
            loss,logits= model(imgs.to(device),labels.to(device))
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
        writer.add_scalar(f'Loss/train', train_loss, epoch)
        writer.add_scalar(f'Acc/train', train_acc, epoch)

        logging.info(f"[Train|{epoch + 1}/{n_epochs}] loss={train_loss:.5f}, acc={train_acc:.5f}")

        # Validation
        model.eval()
        valid_loss, valid_accs = [], []

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                imgs, labels = batch
                # loss,logits = model(imgs.to(device),imgs.size(1),labels.to(device))
                loss, logits = model(imgs.to(device), labels.to(device))
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
            if fold>0:
                torch.save(model.state_dict(), os.path.join(_exp_name,f"best_model_fold{fold + 1}.ckpt"))  # 根据折叠命名
            else:
                model_path=os.path.join(_exp_name, f"best_model_1_1.ckpt")
                if not os.path.exists(_exp_name):
                    os.mkdir(_exp_name)
                torch.save(model.state_dict(), model_path)  # 根据折叠命名
            logging.info(f"Best model found at epoch {epoch + 1}, saving model")
        else:
            stale += 1
            if stale > patience:
                logging.info(f"No improvement for {patience} epochs, early stopping")
                break


if __name__=="__main__":
    set_seed(3704)
    config=Config()
    train_loader,test_loader,speaker_num=get_loader(config.data_dir,config.batch_size,config.n_workers)
    # model=Conformer(num_classes=600,num_attention_heads=4,input_dim=40,num_encoder_layers=4,encoder_dim=512)
    # model=Comformer_net(n_head=4,n_blocks=4,n_spks=600,d_model=512,d_head=256)
#    model = Comformer_net(n_head=2, n_blocks=2, n_spks=600, d_model=512, d_head=256)
#     model = Comformer_net(n_head=1, n_blocks=2, n_spks=600, d_model=512, d_head=256)
    model=Comformer_net(n_head=1, n_blocks=1, n_spks=600, d_model=512, d_head=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    train(model,config.total_steps,300,optimizer,train_loader,test_loader,scheduler=scheduler)



