import torch
import torch.nn.functional as F

def load_embeddings(path="embedding_matrix.pt"):
    data = torch.load(path)
    E = data["weights"]                # [vocab × dim]
    word_to_id = data["word_to_id"]
    id_to_word = data["id_to_word"]
    return E, word_to_id, id_to_word


def search(E, word_to_id, id_to_word, query, k=5):
    if query not in word_to_id:
        return None

    # ID of query word
    qid = word_to_id[query]

    # Vector for query word
    qvec = E[qid]                      # [dim]

    # Normalize everything for cosine similarity
    E_norm = F.normalize(E, dim=1)     # [vocab × dim]
    qvec_norm = F.normalize(qvec, dim=0)  # [dim]

    # Compute cosine similarity with all words
    sims = E_norm @ qvec_norm          # [vocab]

    # Get top-k excluding the query itself
    topk = torch.topk(sims, k + 1).indices.tolist()
    topk = [i for i in topk if i != qid][:k]

    return [(id_to_word[i], float(sims[i])) for i in topk]


def repl():
    E, word_to_id, id_to_word = load_embeddings()

    print("Welcome to Embedding Search!. Type a word to see similare matches or 'quit' to exit the program.")
    print(f"Vocabulary size: {len(word_to_id)} words")
    print()

    while True:
        q = input("> ").strip().lower()
        if q == "quit":
            break

        result = search(E, word_to_id, id_to_word, q)
        if result is None:
            print("Word not in vocabulary.")
            continue

        for word, score in result:
            print(f"{word:20s} {score:.4f}")

        print()


if __name__ == "__main__":
    repl()
