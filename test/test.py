import torch
from torch.utils.data import Dataset, DataLoader

# 1. 自定义一个简单数据集
class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(10).float()     # 特征
        self.y = self.x * 2                   # 标签

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. 创建 Dataset
dataset = MyDataset()

# 3. 创建 DataLoader
loader = DataLoader(dataset, batch_size=3, shuffle=True)

# 4. 使用 DataLoader
for batch_x, batch_y in loader:
    print(batch_x, batch_y)




