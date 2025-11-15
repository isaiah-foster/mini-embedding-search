# Linear Algebra 225 Final Computer Project: Embedding Search

### Description
- This program takes a small corpus of text and embeds it as vectors in a single matrix. The matrix is of size [vocab size x embedding dimension].
- It trains association between words based on pairs of words. The pairs are formed by every word in the text, and every word within two positions, left and right of that word.
- Uses matrix multiplication, matrix transposes, dot products, softmax, and gradient descent to train the model.
- Uses cosine similarity to find words similar to the user's query word to find the n most similar words in the vocabulary after training has happened.


### Try it out yourself!
- To set up your environment, you'll need to install some Python packages. PyTorch is pretty big so i recommend installing it in a virtual environment.
- run ```python -m venv .venv && source .venv/bin/activate``` to create and activate the environment.
- run ```pip install -r requirements.txt``` to install the packages I've included.
- To run the search function, just go to main.py and hit run!
- If you want to train the model yourself, you can go to train.py and run it. Play with the training hyperparameters if you want to see how they change it!

### Assignment Writeup
For this project, I selected option 4, which required writing my own code to implement a linear-algebraic idea. I built a small word-embedding model inspired by the skip-gram architecture used in early natural-language processing systems. The goal of the model was to learn vector representations of words based on the contexts in which they appear. This project let me apply linear algebra directly in software and watch how matrix operations form the basis of modern machine-learning techniques.

My implementation loads a text corpus and converts each word into an integer index, forming a vocabulary. The core of the model is an embedding matrix, where each row corresponds to a word vector in a high-dimensional space. During training, the model takes a “center word” and predicts nearby context words by computing dot products between the center vector and all other word vectors. The dot product is fundamentally a linear-algebraic similarity measure, and the model relies on repeated matrix multiplications and vector operations to compute predictions. These unnormalized scores (called logits in the ML world) are passed through a softmax function to produce a probability distribution across the vocabulary. A cross-entropy loss then measures how well the model predicted the true context word. Gradient descent updates the embedding matrix by adjusting the vectors in the direction that reduces prediction error.

Implementing this algorithm made my understanding of linear algebra’s applications a lot more concrete. I saw how embedding matrices are simply large parameter matrices that get updated through basic operations such as matrix multiplication, transposition, and normalization. Second, the dot product’s geometric meaning (how aligned two vectors are) became obvious when I inspected which words became “nearest neighbors” after training. Words that appeared in similar contexts ended up with vectors that pointed in similar directions, and cosine similarity made this relationship digestible to a human. Watching the embedding space change across training epochs help connect abstract ideas like gradient descent and vector optimization to real numerical changes and performance increases.
