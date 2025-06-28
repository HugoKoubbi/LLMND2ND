class BTENaive:
    def __init__(self):
        self.vocab = {i: chr(i) for i in range(256)}
        self.vocab[256] = '<|endoftext|>'
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        byte_sequence = text.encode('utf-8')
        return [b for b in byte_sequence]  # or use inv_vocab[chr(b)] if mapping needed

    def decode(self, tokens: List[int]) -> str:
        byte_sequence = bytes(tokens)
        return byte_sequence.decode('utf-8')

    def pre_tokenize(self, text: str) -> dict[tuple[str], int]:
        matches = re.finditer(PAT, text)
        tokens = [m.group(0) for m in matches]
        sequences = [self.encode(word) for word in tokens]
        sequence_counts = Counter(sequences)
        return sequence_counts

    def vocab_size(self):
        return len(self.vocab)

    def train(self, texts: List[str], vocab_size: int = 1000):
        all_sequences = []
        for text in texts:
            # Pre-tokenize the text to get sequences
            sequences = self.pre_tokenize(text)        
            pair_counts = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i+1])] += 1
            most_common = max(self.pairs_counts.items(), key=lambda x: (x[1], -tuple(map(ord, x[0]))))
            self.vocab[1+len(self.vocab)] = most_common[0]
            self.inv_vocab[most_common[0]] = 1+len(self.vocab)

    def get_vocab(self) -> dict[int, str]:
        return self.vocab
