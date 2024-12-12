import torch
from torch.utils.data import Dataset

class MaskedDataset(Dataset):
    def __init__(self, data, labels, mask_prob=0.2):
        """
        支持每个epoch内mask固定的自定义数据集。
        
        :param data: 输入特征数据，形状为 (num_samples, num_features)
        :param labels: 标签数据，形状为 (num_samples, ...)
        :param mask_prob: 随机mask的概率，默认为0.2
        """
        self.data = data
        self.labels = labels
        self.mask_prob = torch.tensor(mask_prob, dtype=torch.float32)
        _, self.num_features = self.data.shape

        # 初始化mask矩阵，每行对应一个样本
        self.mask = self._generate_mask()

    def _generate_mask(self):
        """生成mask矩阵"""
        return (torch.rand(self.num_features) > self.mask_prob).float().cuda()

    def refresh_mask(self):
        """刷新每个样本的mask（在新epoch开始时调用）"""
        self.masks = self._generate_mask()

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本及其固定的mask
        :param idx: 样本索引
        :return: mask后的特征和对应标签
        """
        features = self.data[idx, :]
        label = self.labels[idx]

        # 使用预生成的mask
        masked_features = features * self.mask

        return masked_features, label