# Linear Algebra 225 Final Computer Project: Embedding Search

### Description
- This program takes a small corpus of text and embeds it as vectors in a single matrix. The matrix is of size [vocab size x embedding dimension].
- It trains association between words based on pairs of words. The pairs are formed by every word in the text, and every word within two positions left and right of that word.
- Uses matrix multiplication, matrix transposes, dot products, softmax, and gradient descent to train the model.
- Uses cosine similarity to find words similar to the user's query word to find the n most similar words in the vocabulary after training has happened.


### Try it out yourself!
- to set up your environment you'll need to install some python packages. PyTorch is pretty big so i recommend installing it in a virtual environment.
- run ```python -m venv .venv && source .venv/bin/activate``` to create and activate the environment.
- run ```pip install -r requirements.txt``` to install the packages ive included.
- to run the search function just go to main.py and hit run!
- if you want to train the model yourself you can go to train.py and run it. Play with the training hyperparameters if you want to see how they change it!