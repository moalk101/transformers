import re
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TranslationDataSet:
    
    def __init__(self,min_len=5,max_len=64,max_ratio=1.5,limit=None):
        self.WHITELIST = "abcdefghijklmnopqrstuvwxyzäöüß0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥ "
        self.WHITELIST_SET = set(self.WHITELIST.lower()) 
        self.min_ratio = 1 / max_ratio
        self.max_ratio = max_ratio
        self.max_len = max_len
        self.min_len = min_len
        self.data = load_dataset("wmt17", "de-en")
        self.limit = limit if limit != None else len(self.data["train"])
        
    def get_wmt17_datset(self,split="train"):
        return self.data[split].select(range(self.limit))
    
    def _preprocess_text(self,text):
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
        
        text = "".join(c for c in text if c in self.WHITELIST_SET)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_sentence_pair(self,example: dict):
        source_lang = 'de'
        target_lang = 'en'

        source_text = example['translation'][source_lang]
        target_text = example['translation'][target_lang]

        keep_example = True 
        cleaned_source = self._preprocess_text(source_text)
        cleaned_target = self._preprocess_text(target_text)


        source_len = len(cleaned_source.split())
        target_len = len(cleaned_target.split())


        if not (self.min_len <= source_len <= self.max_len and self.min_len <= target_len <= self.max_len):
            keep_example = False

        if source_len > 0 and target_len > 0:
            ratio = source_len / target_len
            if not (self.min_ratio <= ratio <= self.max_ratio):
                keep_example = False
        else:
            keep_example = False

        example['translation'][source_lang] = cleaned_source
        example['translation'][target_lang] = cleaned_target
        example['keep'] = keep_example

        return example
        
        
class TranslationTorchDataset(Dataset):
    def __init__(self,dataset, tokenizer,max_len=64, src_lang="de",tgt_lang="en"):
        super().__init__()
        self.data = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        element = self.data[index]["translation"]
        
        src_ids = self.tokenizer.encode(element[self.src_lang])[:self.max_len]
        tgt_ids = self.tokenizer.encode(element[self.tgt_lang])[:self.max_len]
        
        return {"src": torch.tensor(src_ids,dtype=torch.long),"tgt": torch.tensor(tgt_ids,dtype=torch.long)}
        