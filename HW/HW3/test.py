from models import resnet,resnet18
import torchvision.transforms.v2 as transforms
import torch
from torch.utils.data import DataLoader
from DataSet import FoodDataSet
from utils import get_transforms
import numpy as np
import os
import pandas as pd
def get_models(path):
    # 列出目录中的所有文件
    model_paths = os.listdir(path)
    models = []
    for model_file in model_paths:
        # 结合路径和文件名
        full_path = os.path.join(path, model_file)
        # 初始化模型（假设 resnet() 是模型的构造函数）
        model = resnet18()
        # 加载模型权重并设置为评估模式
        model.load_state_dict(torch.load(full_path))
        model.eval()
        # 将模型添加到列表中
        models.append(model)
    return models
def precision_of_each_class(predictions,labels):
    unique_classes=np.unique(labels)
    class_accuracy={}
    for cls in unique_classes:
        class_indices=np.where(labels==cls)
        correct_predictions=np.sum(predictions[class_indices]==cls)
        total_samples=len(class_indices[0])
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        class_accuracy[cls] = accuracy
    return class_accuracy
def test_with_labels(models,test_loader,augmentations=None):
    test_preds=np.zeros((len(test_data),11))
    flag=True
    labels=[]
    with torch.inference_mode():
        for model_best in models:
            model_best.to(device)
            for i,(data ,label )in enumerate(test_loader):
                preds=model_best(data.to(device)).cpu().numpy()
                if flag:
                    labels.extend(label.tolist())
                test_preds[i*64:64*(i+1),:]+=preds
            flag=False
    return test_preds,labels

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
if __name__=="__main__":
    device="cuda"if torch.cuda.is_available() else "cpu"
    # weights_models=np.array([0.4, 0.4, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5])
    models_from_resnet = get_models("resnet")
    # models_from_models=get_models("models")

    _,test_tfm=get_transforms()
    test_data=FoodDataSet(file_path="food11/test",tfm=_)
    test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
    # test_preds_models,labels=test_with_labels(models_from_models,test_loader)
    test_preds_resnet,_=test_with_labels(models_from_resnet,test_loader)
    # test_preds=test_preds_models*weights_models+test_preds_resnet*(1-weights_models)
    prediction=test_preds_resnet.argmax(axis=1)
    # print( precision_of_each_class(prediction,labels))
    df=pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1,len(test_data)+1)]
    df["Category"] = prediction
    df.to_csv("submission.csv",index = False)