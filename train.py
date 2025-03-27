import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

from model import build_encoder_transformer, build_transformer

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def encoder_transformer_train():
    # data prep: using SciTail Dataset
    dataset = load_dataset("scitail", "snli_format")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def tokenize(data):
        return tokenizer(data["sentence1"], data["sentence2"], padding="max_length", truncation=True)

    def collate(batch):
        return torch.tensor([item["input_ids"] for item in batch])

    tok_dataset = dataset.map(tokenize, batched=True)
    train_dataloader = DataLoader(tok_dataset["train"], batch_size=16, shuffle=True, collate_fn=collate)

    # model setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropouts = {key: 0.1 for key in ['encoder_pe', 'encoder_self_attetion', 'encoder_feed_forward', 'encoder_block']}
    vocab_size = 30522
    seq_len = 512
    encoder_transformer = build_encoder_transformer(vocab_size, seq_len, dropouts).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(encoder_transformer.parameters(), lr=1e-5)

    # training
    epochs = 3
    for epoch in range(epochs):
        encoder_transformer.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            padding_mask = (batch!=0).unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, 1, 1, seq_len)
            padding_mask = padding_mask.repeat(1,8,1,1)
            padding_mask = padding_mask.to(device)
            optimizer.zero_grad()
            out = encoder_transformer(batch, padding_mask)
            loss = criterion(out.view(-1, out.shape[-1]), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader)}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    trained_model_path = os.path.join(current_dir, "trained_encoder_model.pth")
    torch.save(encoder_transformer.state_dict(), trained_model_path)
    print("Training Complete.")
    
    
class CodeDataSet(Dataset):
    def __init__(self, data, tokenizer, seq_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        input_text = "Generate Python: " + example["func_documentation_string"]
        target_text = example["func_code_string"]
        
        input_encoding = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.seq_len, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=self.seq_len, return_tensors="pt")
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }
            
def transformer_train():
    # dataset: CodeSearchNet
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    train = CodeDataSet(dataset["train"], tokenizer)
    valid = CodeDataSet(dataset["validation"], tokenizer)
    
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    source_seq_len = 512
    target_seq_len = 512
    key_list = ['encoder_pe', 'encoder_self_attetion', 'encoder_feed_forward', 'encoder_block', 'decoder_pe', 'decoder_self_attention', 'decoder_cross_attention', 'decoder_feed_forward', 'decoder_block']
    dropouts = {key: 0.1 for key in key_list}
    model = build_transformer(source_vocab_size, target_vocab_size, source_seq_len, target_seq_len, dropouts).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_type_id)
    
    epochs = 3
    num_training_steps = len(train_loader)*epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    
    best_loss = float("inf")
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            
            # masks
            source_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, 1, 1, seq_len)
            source_mask = source_mask.repeat(1,8,1,1)
            source_mask = source_mask.to(device)
            target_mask = (labels!=0).unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, 1, 1, seq_len)
            target_len = labels.shape[1]
            target_no_lookahead_mask = torch.tril(torch.ones((target_len, target_len), device=device)).unsqueeze(0).unsqueeze(0)
            target_mask = target_mask.to(torch.bool) & target_no_lookahead_mask.to(torch.bool)
            target_mask = target_mask.to(device)
            
            out = model(input_ids, labels, source_mask, target_mask)
            loss = loss_fn(out.view(-1, out.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss+=loss.item()
        
        # evaluation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
                
                # masks
                source_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, 1, 1, seq_len)
                source_mask = source_mask.repeat(1,8,1,1)
                source_mask = source_mask.to(device)
                target_mask = (labels!=0).unsqueeze(1).unsqueeze(2).to(device)  # (batch_size, 1, 1, seq_len)
                target_len = labels.shape[1]
                target_no_lookahead_mask = torch.tril(torch.ones((target_len, target_len), device=device)).unsqueeze(0).unsqueeze(0)
                target_mask = target_mask.to(torch.bool) & target_no_lookahead_mask.to(torch.bool)
                target_mask = target_mask.to(device)
                
                out = model(input_ids, labels, source_mask, target_mask)
                loss = loss_fn(out.view(-1, out.shape[-1]), labels.view(-1))
                valid_loss+=loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss = {round(train_loss/len(train_loader), 4)}, Validation Loss = {round(valid_loss/len(valid_loader), 4)}")
        if best_loss > valid_loss:
            best_loss = valid_loss
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "custom_codegen_model.pth")
            torch.save(model.state_dict(), model_path)
            print("Best model updated and saved.")
        
        print()
    print("Training Complete.")
    
    
def prepare_custom_model(tokenizer, seq_len=512, dropouts=None):
    source_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    source_seq_len = seq_len
    target_seq_len = seq_len
    model = build_transformer(source_vocab_size, target_vocab_size, source_seq_len, target_seq_len, dropouts)
    
    return tokenizer, model


def prepare_pretrained_model(model_name="Salesforce/codegen-350M-mono"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return tokenizer, model


def preprocess(example, tokenizer):
    # causal masking internally blocks to know the information of output with the algorithm model can see tokens up to the current token
    prompt = f"Generate Python code: {example['func_documentation_string']}"
    code = example["func_code_string"]
    
    # tokenize, with ratio 150:874 for prompt, code respectively
    prompt_inputs = tokenizer(prompt, truncation=True, max_length=150, padding="max_length")
    code_inputs = tokenizer(code, truncation=True, max_length=874, padding="max_length")

    inputs = {
        "input_ids": prompt_inputs["input_ids"] + code_inputs["input_ids"],
        "attention_mask": prompt_inputs["attention_mask"] + code_inputs["attention_mask"]
    }
    
    # Ensure the total length does not exceed 1024
    inputs["input_ids"] = inputs["input_ids"][:1024]
    inputs["attention_mask"] = inputs["attention_mask"][:1024]

    # Pad to 1024 if necessary
    padding_length = 1024 - len(inputs["input_ids"])
    inputs["input_ids"] += [tokenizer.pad_token_id] * padding_length
    inputs["attention_mask"] += [0] * padding_length

    labels = inputs["input_ids"].copy()  # ~1024 tokens

    # mask the prompt to only generate the coding part
    prompt_len = len(prompt_inputs.input_ids)  # ~150
    labels[:prompt_len] = [-100]*prompt_len
    inputs["labels"] = labels
    
    return inputs

  
def prepare_dataset(model_name="Salesforce/codegen-350M-mono"):
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    tokenizer, model = prepare_pretrained_model(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train = dataset["train"].shuffle(seed=42).select(range(100000)).map(preprocess, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dataset["train"].column_names)
    valid = dataset["validation"].shuffle(seed=42).select(range(20000)).map(preprocess, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dataset["validation"].column_names)
    
    return tokenizer, model, train, valid


def fine_tune_pretrained_model(tokenizer, model, train, valid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir="./codegen-finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=tokenizer
    )
    
    # train
    trainer.train()
    
    # save
    trainer.save_model("./codegen-finetuned")
    tokenizer.save_pretrained("./codegen-finetuned")    
    
    
    
# script
# encoder_transformer_train()
# transformer_train()
tokenizer, model, train, valid = prepare_dataset()
fine_tune_pretrained_model(tokenizer, model, train, valid)