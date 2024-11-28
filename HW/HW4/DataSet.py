import os
import json
import random
from pathlib import  Path
from  torch.utils.data import Dataset
import  torch
class myDataset(Dataset):
    def __init__(self,data_dir,segment_len=128):
        self.data_dir=data_dir
        self.segment_len=segment_len
        mapping_path=Path(data_dir)/"mapping.json"
        mapping=json.load(mapping_path.open())
        self.speaker2id=mapping["speaker2id"]
        metadata_path=Path(data_dir)/"metadata.json"
        metadata=json.load(open(metadata_path))["speakers"]
        self.speaker_num=len(metadata.keys())
        self.data=[]
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"],self.speaker2id[speaker]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        feat_path,speaker=self.data[index]
        mel = torch.load(os.path.join(self.data_dir, feat_path),weights_only=True)
        if len(mel)>self.segment_len:
            start=random.randint(0,len(mel)-self.segment_len)
            mel=torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel=torch.FloatTensor(mel)
        speaker=torch.FloatTensor([speaker]).long()
        return mel,speaker
    def get_speaker_number(self):
        return self.speaker_num

class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path),weights_only=True)

		return feat_path, mel

if __name__=="__main__":
    path="Dataset"
    data=myDataset(path)
    print(data[10])
