import torch
import os
from model import build_encoder_transformer, build_transformer
from transformers import AutoTokenizer, AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "models")


def load_encoder_model():
    vocab_size = 30522
    seq_len = 512
    dropouts = {key: 0.1 for key in ['encoder_pe', 'encoder_self_attetion', 'encoder_feed_forward', 'encoder_block']}
    model = build_encoder_transformer(vocab_size, seq_len, dropouts)

    model_path = os.path.join(model_dir, "trained_encoder_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model


def load_custom_codegen_model(tokenizer, seq_len=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    source_seq_len = seq_len
    target_seq_len = seq_len
    key_list = ['encoder_pe', 'encoder_self_attetion', 'encoder_feed_forward', 'encoder_block', 'decoder_pe', 'decoder_self_attention', 'decoder_cross_attention', 'decoder_feed_forward', 'decoder_block']
    dropouts = {key: 0.1 for key in key_list}
    model = build_transformer(source_vocab_size, target_vocab_size, source_seq_len, target_seq_len, dropouts)
    
    model_path = os.path.join(model_dir, "custom_codegen_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def load_finetuned_codegen_model():
    model_path = os.path.join(model_dir, "codegen-finetuned")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    return tokenizer, model