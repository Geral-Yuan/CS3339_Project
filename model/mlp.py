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
        x = torch.sqrt(x)
        x = self.fc2(x)
        return x
    
    def hidden_layer(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = torch.sqrt(x)
        return x
    
class resNetMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, drop_rate=0.5):
        super(resNetMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        # x = torch.sqrt(x)
        x = self.fc2(x) + x
        x = self.dropout(x)
        x = torch.relu(x)
        # x = torch.sqrt(x)
        x = self.fc3(x)
        return x

class MLPClassifier(ClassifierMixin, BaseEstimator, nn.Module):
    def __init__(self, input_size, output_size, resNet=False, hidden_size=128, drop_rate=0.5, weight_decay=0.0, lr=0.001, batch_size=128, epoch_num=10):
        super(MLPClassifier, self).__init__()
        if resNet:
            self.model = resNetMLP(input_size, output_size, hidden_size, drop_rate)
        else:
            self.model = MLP(input_size, output_size, hidden_size, drop_rate)
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.epoch_num = epoch_num

    def fit(self, train_feature, train_label, val_feature=None, val_label=None):
        label_cnt = []
        for i in range(20):
            label_cnt.append((train_label == i).sum().item())
        
        total_cnt = train_label.shape[0]
        class_weights = [total_cnt / label_cnt[i] for i in range(20)]
        
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).cuda())

        train_feature_tensor = torch.tensor(train_feature, dtype=torch.float32).cuda()
        train_label_tensor = torch.tensor(train_label, dtype=torch.long).cuda()

        dataset = TensorDataset(train_feature_tensor, train_label_tensor)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self.model.cuda().train()

        for epoch in tqdm(range(self.epoch_num)) if val_feature is None or val_label is None else range(self.epoch_num):
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features
                batch_labels = batch_labels

                optimizer.zero_grad()
                output = self.model(batch_features)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
            if val_feature is not None and val_label is not None:
                self.model.eval()
                with torch.no_grad():
                    train_accuracy = self.score(train_feature, train_label)
                    val_accuracy = self.score(val_feature, val_label)
                self.model.train()
                print(f"Epoch {epoch+1}/{self.epoch_num}, train accuracy: {train_accuracy:.6f}, validation accuracy: {val_accuracy:.6f}")

    def predict(self, test_features):
        test_features = torch.tensor(test_features, dtype=torch.float32).cuda()
        self.model = self.model.cuda()

        self.model.eval()
        with torch.no_grad():
            test_output = self.model(test_features)
            _, predicted_labels = torch.max(test_output, 1)

        predicted_labels = predicted_labels.cpu().numpy()
        return predicted_labels

    def score(self, test_features, test_labels):
        predicted_labels = self.predict(test_features)
        accuracy = (predicted_labels == test_labels).mean()
        return accuracy
    
    def hidden_layer(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        self.model = self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.hidden_layer(X)
        return hidden.cpu().numpy()
