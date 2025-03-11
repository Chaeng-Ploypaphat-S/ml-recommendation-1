from models.recommender import RecommenderModel
from data.dataset import Dataset
import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, dataset, num_epochs=10, batch_size=32, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # or any other loss function suitable for your task

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataset.get_batches(batch_size):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataset)}')

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load_data()
    model = RecommenderModel()
    train_model(model, dataset)