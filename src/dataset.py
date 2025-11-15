import re
from collections import Counter

# This file defines a class that processes a text file into pairs of words.
# It visits each word or token in the text and creates pairs with all words n position to the left, and n position to the right.
# n is defined by the window_size parameter and is set to 2 for now.
# e.g., for the sentence "the quick brown fox", with a window size of 2, "quick" would be paired with "the", "brown", and "fox".
# The generated pairs will be sent to the model as indexes in the vocabulary, rather than the raw words.

class SkipGramDataset:
    def __init__(self, corpus_path, window_size=2):
        self.window_size = window_size
        self.tokens = self.load_tokens(corpus_path)
        self.word_to_id, self.id_to_word = self.build_vocab(self.tokens)
        self.pairs = self.generate_pairs(self.tokens)

    def load_tokens(self, path):
        text = open(path).read().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def build_vocab(self, tokens):
        vocab = sorted(set(tokens))
        word_to_id = {w: i for i, w in enumerate(vocab)}
        id_to_word = {i: w for w, i in word_to_id.items()}
        return word_to_id, id_to_word

    def generate_pairs(self, tokens):
        pairs = []
        for i, center in enumerate(tokens):
            for j in range(max(0, i - self.window_size),
                           min(len(tokens), i + self.window_size + 1)):
                if i == j: 
                    continue
                pairs.append((center, tokens[j]))
        return pairs
