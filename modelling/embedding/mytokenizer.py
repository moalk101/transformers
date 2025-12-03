import os
import json
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class MyTokenizer():
    
    def __init__(self, vocab_size=50000, save_dir="my_bpe_tokenizer"):
        self.vocab_size = vocab_size
        self.save_dir = save_dir
        self.tokenizer = None
        
    def train(self,corpus):
        
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]        
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens
        )
        
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        os.makedirs(f"{self.save_dir}", exist_ok=True)
        tokenizer.save(f"{self.save_dir}/tokenizer.json")
        tokenizer.model.save(f"{self.save_dir}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            f"{self.save_dir}",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
        
        
        