import torch
import torch.optim as optim
from torch.utils.data import Dataset


class NeuralEmbedding(torch.nn.Module):
    def __init__(
        self, input_dim=2, output_dim=128, hidden_dims=64, num_hidden=4, sft_beta=4.5
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dims)
        self.fc_hidden = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dims, hidden_dims) for _ in range(num_hidden)]
        )
        self.fc_output = torch.nn.Linear(hidden_dims, output_dim)
        self.softplus = torch.nn.Softplus(beta=sft_beta)

    def forward(self, x):
        h = self.softplus(self.fc1(x))
        for fc in self.fc_hidden:
            h = self.softplus(fc(h))
        return self.fc_output(h)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        learning_rate,
        scheduler=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )

    def train(self, num_epochs=10):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            avg_test_loss = self.evaluate()
            test_losses.append(avg_test_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}"
            )

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)


class TorusDataset(Dataset):
    def __init__(self, toroidal_coords, neural_vectors, transform=None):
        self.toroidal_coords = toroidal_coords
        self.neural_vectors = neural_vectors
        self.transform = transform

    def __len__(self):
        return len(self.toroidal_coords)

    def __getitem__(self, idx):
        sample = self.toroidal_coords[idx]
        neural_vector = self.neural_vectors[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, neural_vector
