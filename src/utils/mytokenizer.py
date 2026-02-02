from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast
import os

class MyTokenizer:
    def __init__(self, vocab_size=50000, save_dir="my_gpt2_bpe"):
        self.vocab_size = vocab_size
        self.save_dir = save_dir
        self.tokenizer = None
        self.pad_token_id = None 
        self.bos_token_id = None
        self.eos_token_id = None

    def train(self, corpus):
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        special_tokens = [
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "[UNK]",
        ]

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
        )

        tokenizer.train_from_iterator(corpus, trainer=trainer)

        os.makedirs(self.save_dir, exist_ok=True)
        tokenizer.save(f"{self.save_dir}/tokenizer.json")

        self.tokenizer = GPT2TokenizerFast(
            tokenizer_file=f"{self.save_dir}/tokenizer.json",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def load(self, path):
        self.tokenizer = GPT2TokenizerFast(
            tokenizer_file=f"{path}/tokenizer.json",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        return self
        
    def encode(self, text,add_special_tokens=True):
        ids = self.tokenizer.encode(text)
        if add_special_tokens:
            ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
        return ids
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
        
        
        