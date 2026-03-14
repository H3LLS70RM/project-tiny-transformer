import torch

def decode(tokens, itos):
        return ''.join([itos[token.item()] for token in tokens if token.item() != 0])  # skip padding

def build_char_tokenizer(dataset, max_len):
        all_text = ''.join(dataset)
        vocab = sorted(set(all_text))
        stoi = {ch: i+1 for i, ch in enumerate(vocab)}  # 0 for padding
        itos = {i+1: ch for i, ch in enumerate(vocab)}
        stoi['<pad>'] = 0
        itos[0] = '<pad>'

        def tokenize(s):
            return [stoi.get(ch, 0) for ch in s]

        def batch_tokenize(batch):
            tokenized = [tokenize(seq)[:max_len] for seq in batch]
            padded = [seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq for seq in tokenized]
            tensor = torch.tensor(padded, dtype=torch.long)
            inp = tensor[:, :-1]
            tgt = tensor[:, 1:]
            return inp, tgt

        return batch_tokenize, stoi, itos