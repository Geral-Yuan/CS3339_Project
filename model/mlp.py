import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import ClassifierMixin,BaseEstimator
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, drop_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class MLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, input_size, output_size, hidden_size=128, drop_rate=0.5, weight_decay=0.0, lr=0.001, epoch_num = 10):
        super(MLPClassifier, self).__init__()
        self.model = MLP(input_size, output_size, hidden_size, drop_rate)
        self.weight_decay = weight_decay
        self.lr = lr
        self.epoch_num = epoch_num

    def fit(self, train_feature, train_label):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_feature_tensor = torch.from_numpy(train_feature).float()
        train_label_tensor = torch.from_numpy(train_label).long()

        dataset = TensorDataset(train_feature_tensor, train_label_tensor)

        batch_size = 256
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = self.model.cuda().train()

        for epoch in tqdm(range(self.epoch_num)):
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.cuda()
                batch_labels = batch_labels.cuda()

                optimizer.zero_grad()
                output = self.model(batch_features)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

    def predict(self, test_features):
        test_feature_tensor = torch.from_numpy(test_features).float().cuda()
        self.model = self.model.cuda()

        self.model.eval()
        with torch.no_grad():
            test_output = self.model(test_feature_tensor)
            _, predicted_labels = torch.max(test_output, 1)

        predicted_labels = predicted_labels.cpu().numpy()
        return predicted_labels

    def score(self, test_features, test_labels):
        predicted_labels = self.predict(test_features)
        accuracy = (predicted_labels == test_labels).mean()
        return accuracy
