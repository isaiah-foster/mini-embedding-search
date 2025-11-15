import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset import SkipGramDataset
from model import SkipGramEmbedding

#This file contains the training loop for my skip-gram embedding model defined in model.py
# It uses the dataset defined in dataset.py to load text data and generate training pairs.

# test the output of my dataset class
dataset = SkipGramDataset(corpus_path='src/corpus.txt', window_size=2)
print(f'Vocabulary size: {len(dataset.word_to_id)}')
print(f'Number of training pairs: {len(dataset.pairs)}')
print(f'Total tokens: {len(dataset.tokens)}')


# put all pairs into a PyTorch Dataset for batching. This just converts words to their corresponding indexes in the vocab
# and returns them as tensors compatible with PyTorch's DataLoader
class PairDataset(Dataset):
    def __init__(self, pairs, word_to_id):
        self.data = [(word_to_id[c], word_to_id[t]) for c, t in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


def train():
    # Hyperparameters
    embedding_dim = 64
    batch_size = 128
    epochs = 5
    lr = 0.01

    # Load dataset
    data = SkipGramDataset("src/corpus.txt", window_size=2)
    vocab_size = len(data.word_to_id)

    dataset = PairDataset(data.pairs, data.word_to_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model + optimizer + loss
    model = SkipGramEmbedding(vocab_size, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # most important and linear algebra heavy part of the training loop
    # This softmaxes all logits for each center word across the vocabulary
    # turns a messy vector of unnormalized scores into a proper probability distribution
    # turns negatives into small positives and large positives into values close to 1
    # 
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0

        for center_ids, target_ids in dataloader:
            optimizer.zero_grad()

            logits = model(center_ids)                  # [batch × vocab]
            loss = criterion(logits, target_ids)        # cross-entropy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save embeddings
    torch.save({
        "weights": model.embedding_matrix.weight.detach(),
        "word_to_id": data.word_to_id,
        "id_to_word": data.id_to_word
    }, "embedding_matrix.pt")

    print("Training complete. Embeddings saved to embeddings.pt")


if __name__ == "__main__":
    train()
