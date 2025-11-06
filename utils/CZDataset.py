from torch.utils.data import Dataset

class CZDataset(Dataset):
    def __init__(self, data, labels, sampleID=None):
        self.data = data
        self.labels = labels
        if sampleID != None:
            self.sampleID = sampleID
        else:
            self.sampleID = None
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.sampleID != None:
            return self.data[idx], self.labels[idx], self.sampleID[idx]
        else:
            return self.data[idx], self.labels[idx]
