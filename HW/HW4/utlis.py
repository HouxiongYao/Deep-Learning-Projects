import numpy as np
import torch
import random
import math

from numpy.ma.core import append
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import random_split,DataLoader
from DataSet import myDataset
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
from tqdm.auto import tqdm
import csv
def parse_args():
	"""arguments"""
	config = {
		"data_dir": "./Dataset",
		"save_path": "model.ckpt",
		"batch_size": 32,
		"n_workers": 8,
		"valid_steps": 2000,
		"warmup_steps": 1000,
		"save_steps": 10000,
		"total_steps": 70000,
	}
	return config
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
def collate_batch(batch):
    mel,speaker=zip(*batch)
    mel=pad_sequence(mel,batch_first=True,padding_value=-20)
    #使得mel中每个元素的长度一致
    #mel（batch_size,length,feature）
    return mel,torch.FloatTensor(speaker).long()
def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)
	return feat_paths, torch.stack(mels)
def get_loader(data_dir,batch_size,n_workers):
    dataset=myDataset(data_dir)
    speaker_num=dataset.get_speaker_number()
    train_len=int(0.9*len(dataset))
    lengths=[train_len,len(dataset)-train_len]
    trainset,validset=random_split(dataset,lengths)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return train_loader,valid_loader,speaker_num
def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def test(
	models,
	dataloader,
):
	output_path="./output.csv"
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")
	results = [["Id", "Category"]]
	data_dir = "./Dataset"
	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())
	outs=torch.zeros(len(dataloader),600)
	all_feat_path=[]
	append_flag=True
	for model_pair in models:
		model, model_path = model_pair
		model.load_state_dict(torch.load(model_path, weights_only=True))
		model.eval()
		model.to(device)
		for i,(feat_paths, mels) in tqdm(enumerate(dataloader)):
			with torch.no_grad():
				if append_flag:
					all_feat_path.append(feat_paths[0])
				mels = mels.to(device)
				out = model(mels).cpu()
				outs[i]=outs[i]+out
		append_flag=False
	preds = outs.argmax(1).numpy()
	for feat_path, pred in zip(all_feat_path, preds):
		results.append([feat_path, mapping["id2speaker"][str(pred)]])
	#
	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)
