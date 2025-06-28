import regex as re
from collections import Counter
import sys
from typing import List, Tuple
print(sys.executable)
print(sys.version)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BTENaive:
    def __init__(self):
        # Initialize vocab with all single bytes as tokens (tuple of one int)
        self.vocab = {i: (i,) for i in range(256)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = []  # List of merges as tuples of bytes, e.g. (97, 98)

    def pre_tokenize(self, text: str) -> List[Tuple[int, ...]]:
        # Convert text to list of singleton byte tuples
        #byte_seq = text.encode('utf-8')
        matches = re.finditer(PAT, text)
        words = [m.group(0) for m in matches]
        sequence_2 = [word.encode('utf-8') for word in words]
        seq_counts= Counter(sequence_2)
        return seq_counts

    def train(self, texts: List[str], vocab_size: int = 1000):
        # Pre-tokenize and count sequences
        sequences=self.pre_tokenize(texts)
        while len(self.vocab) < vocab_size:
            pair_counts = Counter()
            for seq, count in sequences.items():
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i+1])
                    pair_counts[pair] += count
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda x: (x[1], -x[0][0], -x[0][1]))[0]

            # Add new merged token
            new_id = max(self.vocab.keys()) + 1
            print(new_id, best_pair)
            self.vocab[new_id] = best_pair
            self.inv_vocab[best_pair] = new_id
            self.merges.append(best_pair)
            # Update seq_counts to reflect the new merged token
            new_seq = (new_id,)
            for seq in list(sequences.keys()):
                if best_pair in zip(seq, seq[1:]):
                    new_seq = tuple(b if b not in best_pair else new_id for b in seq)
                    sequences[new_seq] += sequences[seq]
                    del sequences[seq]

            # TODO: update seq_counts replacing occurrences of best_pair with merged token

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode('utf-8'))
        # Apply merges in order
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == (pair[0],) and tokens[i+1] == (pair[1],):
                    tokens[i] = pair
                    del tokens[i+1]
                else:
                    i += 1
        # Convert to token IDs
        return tokens
    def decode(self, token_ids: List[int]) -> str:
        bytes_seq = []
        for tid in token_ids:
            token = self.vocab[tid]
            if isinstance(token, int):
                bytes_seq.append(token)
            else:
                bytes_seq.extend(token)
        return bytes(bytes_seq).decode('utf-8', errors='replace')
    def vocab_size(self) -> int:
        return len(self.vocab)

text="bla bla bla bo bo bo"
print(list(text.encode('utf-8')))
matches = re.finditer(PAT, text)
liste= [m.group(0) for m in matches]
sequences = [tuple(c for c in word if c != ' ') for word in liste]
sequence_counts = Counter(sequences)
print(sequence_counts)
print(sequences)
            
        
if __name__ == "__main__":
    tokenizer = BTENaive()

    text = "bla bla bla bo bo bo"
    print(tokenizer.encode(text))
    print("Encoded:", print(tokenizer.encode(text)))
    print("Decoded:", print(tokenizer.decode(tokenizer.encode(text))))

    print("\nBefore training vocab size:", tokenizer.vocab_size())
    tokenizer.train([text], vocab_size=300)
    print("After training vocab size:", tokenizer.vocab_size())

    print("\nNew tokens in vocab (last 5):")
    for k, v in list(tokenizer.vocab.items())[-5:]:
        print(f"{k}: {v}")



    
