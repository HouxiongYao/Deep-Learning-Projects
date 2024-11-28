from utlis import test
from pathlib import Path
from utlis import  inference_collate_batch
from DataSet import InferenceDataset
from  torch.utils.data import  DataLoader
from Conformer import Comformer_net
base_dir=Path('models')
data_dir="./Dataset"
model1_path=base_dir/'best_model.ckpt'
model2_path=base_dir/'best_model_1_1.ckpt'
model3_path=base_dir/'best_model_1_2.ckpt'
model4_path=base_dir/'best_model_2_2.ckpt'
model1 = Comformer_net(n_head=4, n_blocks=4, n_spks=600, d_model=512, d_head=256)
model2 = Comformer_net(n_head=1, n_blocks=1, n_spks=600, d_model=512, d_head=256)
model3 = Comformer_net(n_head=1, n_blocks=2, n_spks=600, d_model=512, d_head=256)
model4 = Comformer_net(n_head=2, n_blocks=2, n_spks=600, d_model=512, d_head=256)
dataset = InferenceDataset(data_dir)
# testset=[dataset[i] for i in range(10)]
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=inference_collate_batch,
)
models=[(model1,model1_path),(model2,model2_path),(model3,model3_path),(model4,model4_path)]
test(models,dataloader)