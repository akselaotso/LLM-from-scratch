import re

class simpleTokenizerClass:
    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def update_vocab(self, new_vocab: dict):
        self.str_to_int = new_vocab
        self.int_to_str = {i:s for s, i in new_vocab.items()}

    def construc_vocab(self, tokens):
        tokens = sorted(set(tokens))
        tokens.extend("<|endoftext|>", "<|unk|>")
        vocab = {word:number for number, word in enumerate(tokens)}

        return vocab

    def tokenize(self, text, keepWhitespaces = True):
        text = re.split(r'([,.?!;-:_()"\'\[\]{}]|\n|\s)', text)

        if keepWhitespaces == False:
            text = [item for item in text if item.strip()]

        text = [item if item in self.str_to_int else "<|unk|>" for item in text]

        return text
    
    def new_vocab_from_text(self, text, keepWhitespaces = True):
        tokens = self.tokenize(text, keepWhitespaces=keepWhitespaces)
        vocab = self.construc_vocab(tokens)
        self.update_vocab(new_vocab=vocab)

    def encode(self, text, keepWhitespaces = True):
        ids = [self.str_to_int[i] for i in self.tokenize(text, keepWhitespaces=keepWhitespaces)]
        return ids

    def decode(self, ids, add_whitespaces = False):
        joiner = "" if not add_whitespaces else " "
        
        text = joiner.join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) if add_whitespaces else text

        return text
    
