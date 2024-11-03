
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import  Dataset
class FoodDataSet(Dataset):
    def __init__(self, tfm,file_path=None, files=None):
        super().__init__()
        if file_path is None and files is None:
            raise ArithmeticError("No Inputs")
        if files is not None:
            self.files=files
        else:
            self.path=file_path
            self.files=sorted([os.path.join(file_path, x) for x in os.listdir(file_path) if x.endswith(".jpg")])
        self.transform=tfm
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        filename=self.files[idx]
        im=Image.open(filename)
        im=self.transform(im)
        try:
            filename = os.path.basename(filename)
            label=int(filename.split("_")[0])
        except (ValueError, IndexError) as e:
            label = -1  # 处理转换失败或索引错误
        return im,label
if __name__=="__main__":
    path="food11/training"
    test_tfm=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    Food=FoodDataSet(path,test_tfm)



