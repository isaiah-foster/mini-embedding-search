import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import SkipGramDataset

# this file defines my embedding model that will train on the dataset defined in dataset.py


# embedding model for skip-gram
class SkipGramEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dimension):
        super().__init__()
        # nn from PyTorch creates a matrix of size [vocab_size × embedding_dim]
        # each row represents a word in the vocab as a vector of size embedding_dimension
        # This puts each vector in a vector space where its position relates to all other words in the vocab
        # e.g., "king" might be close to "queen", "prince", "royal", etc. by cosine similarity
        # The vectors are initialized randomly to small numbers by PyTorch and learned during training, just like in an LLM
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_dimension)

    def forward(self, center_ids):

        # Create a new matrix of vector embedding rows by looking up the embedding vectors for the center words
        # the size of the matrix v_center will be [batch_size × embedding_dimension]
        # The input center_ids is a collection of indexes in the embedding matrix
        # The output v_center is the corresponding embedding vectors for those indexes
        # These groups of vectors will be used to train the model to shift the embeddings in the right direction
        v_center = self.embedding_matrix(center_ids)


        # This line performs a matrix multiplication between the center word embeddings found earlier and the transpose of the embedding matrix
        # The result is a matrix of shape [batch_size × vocab_size]
        # Each row corresponds to a center word, and each each of its members tells us how similar that word is to the word represented by that column index
        # The values in this matrix (logits) represent the unnormalized scores for each*
        # This effectively computes the dot product between each center word vector and every other word vector in the vocabulary
        # showing how similar they are in the embedding space
        logits = v_center @ self.embedding_matrix.weight.T
        #embedding_matrix itself is an object, and .weight gives us access to the actual matrix data representing the embeddings
        return logits
