from collections import Counter
import regex as re
import pickle


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# convert to a list of integers
def get_stats(tokens):
    pairs={}
    for pair in zip(tokens, tokens[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs 
def get_pair(tokens):
    pair = get_stats(tokens)
    top_pair = max(pair, key=pair.get)
    return top_pair
def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def merge_tokens(tokens, pair, pair_id):
    new_tokens= []
    i = 0
    while i < len(tokens):
        if i< len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
            new_tokens.append(pair_id)
            i += 2  # Skip the next token since it's part of the pair
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def toy_model_bpe(text, n_vocab):
    merge= {}
    vocab=[i for i in range(256)]
    inv_vocab = [i for i in range(256)]  # Inverse vocabulary mapping
    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens))  # Convert to a list of integers
 # Initial vocabulary with single byte tokens
    while len(vocab)< n_vocab:
        pair = get_pair(tokens)
        if not pair:
            break
        pair_id = max(vocab) + 1  # New token ID
        vocab.append(pair_id)
        inv_vocab.append(pair)  # Add new token to vocabulary
        merge[pair]= pair_id  # Store the merge operation
        tokens = merge_tokens(tokens, pair, pair_id)  # Merge tokens in the text
    return vocab, inv_vocab, tokens, merge

def compression_ratio(original_text, compressed_text):
    original_size = len(original_text.encode('utf-8'))
    compressed_size = len(compressed_text)
    return compressed_size / original_size

def decode(tokens, inv_vocab):
    tokens=b''.join(inv_vocab[t] for t in tokens)
    return tokens.decode('utf-8', errors='replace')

def encode_naive(text, merges):
    tokens= text.encode('utf-8')
    tokens = list(map(int, tokens))  # Convert to a list of integers
    for (p0, p1), i in merges.items():
        # Merge tokens in the text
        pair = (p0, p1)
        i = 0
        while i < len(tokens)-1:
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                tokens[i] = i  # Replace the pair with its merged token ID
                del tokens[i + 1]
                i+= 2
            else:
                i += 1
    return tokens
def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def encode(text, merges):
    tokens=list(text.encode('utf-8'))
    while len(tokens) > 1:
        stats = get_stats(tokens)
        pair  = min(stats, key=lambda p: merges.get(p,float("inf")))
        if pair not in merges:
            break
        idx=merges[pair]
        tokens = merge_tokens(tokens, pair, idx)
    return tokens
 
class BasicTokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}

    def train(self, text, vocab_size, verbose=False):
        tokens = list(text.encode('utf-8'))
        while len(self.vocab) < vocab_size:
            pair = get_pair(tokens)
            if not pair:
                break
            pair_id = max(self.vocab.keys()) + 1
            self.vocab[pair_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.inv_vocab[pair] = pair_id
            self.merges[pair]= pair_id
            tokens = merge_tokens(tokens, pair, pair_id)
            if verbose:
                print(f"Pair: {pair}, ID: {pair_id}")
        return tokens
    
    def merge_tokens(self, tokens, pair, pair_id):
        new_tokens= []
        i = 0
        while i < len(tokens):
            if i< len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(pair_id)
                i += 2  # Skip the next token since it's part of the pair
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def get_stats(self, tokens):
        pairs={}
        for pair in zip (tokens, tokens[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs 
    
    def get_pair(self, tokens):
        pair = get_stats(tokens)
        top_pair = max(pair, key=pair.get)
        return top_pair
    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) > 1:
            stats = get_stats(tokens)
            pair  = min(stats, key=lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx=self.merges[pair]
            tokens = merge_tokens(tokens, pair, idx)
        return tokens
    
    def decode(self, tokens):
        tokens=b''.join(self.vocab[t] for t in tokens)
        return tokens.decode('utf-8', errors='replace')

class RegexTokenizer(BasicTokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
    
# Example usage

if __name__ == "__main__":
    tokenizer = BasicTokenizer()
    text = "hello worldjjejdjdjdjdndakdnfaz;,nnbqzjtgbvnBVKLJ<HZBAJ"
    vocab_size = 260
    tokens = tokenizer.train(text, vocab_size, verbose=True)
    print("Tokens:", tokens)
    
    encoded_text = tokenizer.encode(text)
    print("Encoded Text:", encoded_text)
    
    decoded_text = tokenizer.decode(encoded_text)
    print("Decoded Text:", decoded_text)
    
    tokenizer = RegexTokenizer()
    text = "hello world! This is a test. Let's see how it works."
    tokenizer.encode_ordinary(text)
    print("Ordinary Encoded Text:", tokenizer.encode_ordinary(text))
    tokenizer.train(text, vocab_size=259, verbose=True)

    with open("TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = list(text.encode("utf-8"))  # returns a list of integers (0–255)
    block_size = 128
    input_ids = tokenizer.encode(text)
    # split into chunks
    chunks = [input_ids[i:i+block_size] for i in range(0, len(input_ids) - block_size + 1, block_size)]
    tokenizer_naive =BasicTokenizer()
    tokenizer = RegexTokenizer()
    tokenizer_naive.train(text, vocab_size=300, verbose=True)
    tokenizer.train(text, vocab_size=300, verbose=True)
    with open("my_tokenizer_naive.pkl", "wb") as f:
        pickle.dump(tokenizer_naive, f)
    with open("my_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

#text ='ARPE- Avant de terminer lARPE demander une lettre de bilan à l’encadrant. Cette lettre serait a m’envoyer directement. - Finalisation du rapport final, à envoyer avant le 5 septembre 2025. Pour le rapport final, on s’attend à quelque chose de plutôt propre avec une page de garde contenant: titre, encadrants, date, logos des universités.Il sagit dun rapport scientifique, qui pourrait contenir un article, si votre travail a abouti à une publication. Mais inclure aussi une introduction générale visant à un public non spécialiste sur le sujet, et une section de conclusion apportant un point de vue plus personnel clarifiant ce qui vous a apporté cette année.- Résumé étendu (auto contenu aussi pour le 5/9): ceci est distillé du rapport. Un document de 6 à 8 pages (références incluses). Une sorte de mini-article qui synthétise le projet et les contributions. Mais moins technique et plus orienté à la divulgation. - Préparation de la soutenance. La date tentative est le vendredi 12 septembre de 14 h à 17 h.' 
#v,inv_vocab,t,m=toy_model_bpe(text, 350)
#print(encode("hello world", m))
#print("Merges:", m)
#inv_vocab2 = {i: bytes([i]) for i in range(256)}  # Inverse vocabulary mapping
#for (p0, p1),i in m.items():
#    inv_vocab2[i] = inv_vocab2[p0]+inv_vocab2[p1]  # Add merged tokens

#print("compression ratio:", compression_ratio(text, t))
#decoded_text = decode(t, inv_vocab2)
#print("Decoded Text:", decoded_text)
