class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return scores

    def train_model(self, train_loader, optimizer, criterion, num_epochs):
        for epoch in range(num_epochs):
            for user_ids, item_ids, ratings in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(user_ids, item_ids)
                loss = criterion(outputs, ratings)
                loss.backward()
                optimizer.step()