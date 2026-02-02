from src.modelling.model.transformer import Transformer
from torch.utils.data import DataLoader
from src.utils.dataset import TranslationDataSet, TranslationTorchDataset
from src.utils.mytokenizer import MyTokenizer
from src.utils.learning_rate import TransformerLR
from src.utils.batch_sampler import TokenBatchSampler
import torch.nn as nn
import torch
import os
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import sacrebleu
from tqdm import tqdm





def corpus_iterator(dictionary):
    for ex in dictionary:
        yield ex["translation"]["de"]
        yield ex["translation"]["en"]


def collate_fn(batch, pad_id):
    srcs = [b["src"] for b in batch]
    tgts = [b["tgt"] for b in batch]

    src_batch = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgt_batch = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)

    src_mask = (src_batch != pad_id)
    tgt_padding_mask = (tgt_batch != pad_id)

    return src_batch, tgt_batch, src_mask.long(), tgt_padding_mask.long()

class CollateWithPad:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        return collate_fn(batch, self.pad_id)
    
def compute_lengths(dataset):
    return [len(item["src"]) + len(item["tgt"]) for item in dataset]

def run_validation(model, val_loader, loss_fn, device,limit=-1):
    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (src, tgt, src_mask, tgt_padding_mask) in enumerate(val_loader):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            tgt_padding_mask = tgt_padding_mask.to(device, non_blocking=True)

            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            tgt_mask_input = tgt_padding_mask[:, :-1]

            logits = model(
                src,
                decoder_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask_input
            )

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                decoder_target.reshape(-1)
            )

            total_val_loss += loss.item()
            num_batches += 1
            if batch_idx == limit:
                break

    model.train()
    return total_val_loss / num_batches

def compute_bleu(
    model,
    val_loader,
    tokenizer,
    device,
    max_length=50,
    max_batches=None 
):
    model.eval()

    hypotheses = []
    references = []

    with torch.no_grad():
        for i, (src, tgt, src_mask, _) in enumerate(
            tqdm(val_loader, desc="Computing BLEU", leave=False)
        ):
            if max_batches is not None and i >= max_batches:
                break

            src = src.to(device)
            tgt = tgt.to(device)

            pred = model.generate(
                src,
                max_length=max_length
            )

            for p, r in zip(pred, tgt):
                p = [
                    t.item() for t in p
                    if t.item() not in {
                        0,
                        1,
                        2
                    }
                ]
                r = [
                    t.item() for t in r
                    if t.item() not in {
                        0,
                        1,
                        2
                    }
                ]

                hypotheses.append(tokenizer.decode(p))
                references.append(tokenizer.decode(r))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score



def main():
    
    ds = TranslationDataSet(limit=1500000)
    raw = ds.get_wmt17_datset()
    cleaned = raw.map(ds.clean_sentence_pair)
    cleaned = cleaned.filter(lambda x: x["keep"]).select(range(1200000))
    # tokenizer = MyTokenizer(save_dir=r"C:\Users\modar\Desktop\Uni\transforemer\models\WMTBPETokenizer")
    # iterator = corpus_iterator(cleaned)
    # tokenizer.train(iterator)
    tokenizer = MyTokenizer().load(r"C:\Users\modar\Desktop\Uni\transforemer\models\WMTBPETokenizer")
    pad_id = tokenizer.tokenizer.pad_token_id
    torchdataset = TranslationTorchDataset(cleaned,tokenizer)
    # loader = DataLoader(torchdataset,batch_size=32,shuffle=True,collate_fn=lambda b: collate_fn(b, pad_id))
    model = Transformer(50000,512,8,6,6,2048,0.1,64)

    train_size = 1000000
    val_size = 200000

    # Split the dataset
    if os.path.exists(r"C:\Users\modar\Desktop\Uni\transforemer\data\train_indices.pt"):
        print("loading existing splits")
        val_indices = torch.load(r"C:\Users\modar\Desktop\Uni\transforemer\data\val_indices.pt")
        train_indices = torch.load(r"C:\Users\modar\Desktop\Uni\transforemer\data\train_indices.pt")
        val_dataset = Subset(torchdataset, val_indices)
        train_dataset = Subset(torchdataset, train_indices)
    else:
        train_dataset, val_dataset = random_split(torchdataset, [train_size, val_size])
        torch.save(val_dataset.indices, r"C:\Users\modar\Desktop\Uni\transforemer\data\val_indices.pt")
        torch.save(train_dataset.indices, r"C:\Users\modar\Desktop\Uni\transforemer\data\train_indices.pt")
    collator = CollateWithPad(pad_id)

    # lengths = compute_lengths(train_dataset)

    # sampler = TokenBatchSampler(
    #     lengths=lengths,
    #     max_tokens=10000,  
    #     shuffle=True
    # )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=128,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1,betas=(0.9,0.98),eps=1e-9)
    scheduler = TransformerLR(optimizer, d_model =model.d_model, warmup_steps=4000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start_epoch = 0
    current_loss = 10000
    global_step = 0
    if os.path.exists(r"C:\Users\modar\Desktop\Uni\transforemer\models\Transformermodel\v4\checkpoint.pt"):
      ckpt = torch.load(r"C:\Users\modar\Desktop\Uni\transforemer\models\Transformermodel\v4\checkpoint.pt")
      model.load_state_dict(ckpt["model_state"])
      optimizer.load_state_dict(ckpt["optimizer_state"])
      start_epoch = ckpt["epoch"] + 1
      current_loss = ckpt["loss"]
      global_step = ckpt["global_step"]
    
    
    num_epochs = 5
    writer = SummaryWriter(log_dir=r"C:\Users\modar\Desktop\Uni\transforemer\models\Transformermodel\v4\log")
    best_val_loss = 999999
    best_globsl_val_loss = 99999
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (src, tgt, src_mask, tgt_padding_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            src = src.to(device,non_blocking=True)
            tgt = tgt.to(device,non_blocking=True)
            src_mask = src_mask.to(device,non_blocking=True)
            tgt_padding_mask = tgt_padding_mask.to(device,non_blocking=True)

            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            tgt_mask_input = tgt_padding_mask[:, :-1]

            logits = model(
                src,
                decoder_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask_input
            )

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                decoder_target.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            # if loss.item() < current_loss:
            #     print("loss got better")
            #     current_loss = loss.item()
            #     checkpoint = {
            #         "epoch": epoch,
            #         "model_state": model.state_dict(),
            #         "optimizer_state": optimizer.state_dict(),
            #         "loss": current_loss,
            #         "global_step" : global_step
            #     }
            #     torch.save(checkpoint, r"C:\Users\modar\Desktop\Uni\transforemer\models\Transformermodel\v4\checkpoint.pt")

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx+1} Loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), global_step)
                print("run validation")
                val_loss = run_validation(model, val_loader, loss_fn, device,limit=2)
                print("compute bleu")
                bleu = compute_bleu(model,val_loader,tokenizer,"cuda",max_batches=1)
                if val_loss < best_val_loss:
                    print("validation loss got improved!")
                    checkpoint = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loss": current_loss,
                        "global_step" : global_step
                    }
                    best_val_loss = val_loss
                    torch.save(checkpoint, r"C:\Users\modar\Desktop\Uni\transforemer\models\Transformermodel\v4\checkpoint_best_val_loss.pt")
                writer.add_scalar("Loss/val", val_loss, global_step)
                writer.add_scalar("BLEU",bleu,global_step)
                print(f"Epoch {epoch} Batch {batch_idx+1} Validation Loss: {val_loss:.4f}")
                print(f"Epoch {epoch} Batch {batch_idx+1} BLEU Score: {bleu:.4f}")
            global_step += 1
        
        print(f"Epoch completed. Running validation on entire dataset!")
        globsl_val_loss = run_validation(model, val_loader, loss_fn, device)
        if globsl_val_loss < best_globsl_val_loss:
            print("validation loss got improved!")
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": current_loss,
                "global_step" : global_step
            }
            best_globsl_val_loss = globsl_val_loss
            torch.save(checkpoint, r"C:\Users\modar\Desktop\Uni\transforemer\models\Transformermodel\v4\checkpoint_best_val_loss.pt")
        writer.add_scalar("Loss/val_all", globsl_val_loss, global_step)
        print(f"Val Loss: {globsl_val_loss:.4f}")
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    writer.close()
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()