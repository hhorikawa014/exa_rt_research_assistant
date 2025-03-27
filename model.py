import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float | None):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)  # positional encoding blueprint matrix
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # numerator in the formula, shape=(seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model)) # for numerical stability using log space
        pe[:, 0::2] = torch.sin(position*div_term)  # at 0, 2, ..., 2k, ...
        pe[:, 1::2] = torch.cos(position*div_term)  # at 1, 3, ..., 2k+1, ...
        self.pe = pe.unsqueeze(0).to(device)  # shape=(1, seq_len, d_model)
        
        #self.register_buffer('pe', pe.unsqueeze(0))  # save the positional encoding in the module along with the saved model (NOT as a learned param)
        
    def forward(self, x):
        if self.dropout is not None:
            return self.dropout(x+(self.pe[:, :x.shape[1], :].to(x.device)).requires_grad_(False))  # setting requries_grad_ False ensures pe is not learned
        return x+(self.pe[:, :x.shape[1], :].to(x.device)).requires_grad_(False)

# attention is all you need!!
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float | None):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model%h == 0, "d_model has to be divisible by h"
        
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        
        self.attention = ScaledDotProductAttention(self.d_k, self.dropout)
        
    def forward(self, Q, K, V, mask=None):
        # shape transition: Q,K,V=(batch_size, seq_len, d_model) -W-> (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -T-> (batch_size, h, seq_len, d_k)
        q = self.w_q(Q).view(Q.shape[0], Q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = self.w_k(K).view(K.shape[0], K.shape[1], self.h, self.d_k).transpose(1, 2)
        v = self.w_v(V).view(V.shape[0], V.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x_out, attn = self.attention(q,k,v,mask)
        # x_out=(batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x_out = x_out.transpose(1, 2).contiguous().view(x_out.shape[0], -1, self.h*self.d_k)  # contiguous to ensure the tensor is stored as contiguous blocks
        return self.w_o(x_out)  # -> (batch_size, seq_len, d_model)
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout: nn.Dropout | None):
        super().__init__()
        self.d_k = d_k
        self.dropout = dropout
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.d_k)  # (batch_size, h, seq_len, d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e-9)
        if self.dropout is not None:
            scores = self.dropout(scores)
        attn = F.softmax(scores, dim=-1)  # -> (batch_size, h, seq_len, seq_len)
        return torch.matmul(attn, V), attn  # -> (batch_size, h, seq_len, d_k)


class LayerNormalization(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))  # multiplier
        self.beta = nn.Parameter(torch.zeros(1))  # bias
        
    def forward(self, x):
        mean = x.float().mean(dim=-1, keepdim=True)
        std = x.float().std(dim=-1, keepdim=True)
        return self.gamma*(x-mean)/(std+self.epsilon)+self.beta
        
        
class FeedForward(nn.Module):
    # formula: max(0, xW1+b1)W2+b2, where W1,2 are linear layers, b1,b2 are biases, and max(0, z) is done by a relu layer
    def __init__(self, d_model: int, d_ff: int, dropout: float | None):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.dropout is not None:
            return self.linear2(self.dropout(self.relu(self.linear1(x))))
        return self.linear2(self.relu(self.linear1(x)))
    

# skip connection norm->norm aside from norm->feedforward
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float | None):
        super().__init__()
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):  # sublayer is prev layer
        if self.dropout is not None:
            return self.dropout(sublayer(self.norm(x)))
        return sublayer(self.norm(x))
    

# for the customizability (of dropout), pass each block to stack up instead of parameters to create blocks from scratch
class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float | None):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, source_mask):
        x_out = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, source_mask))  # forward in MultiHeadAttention
        x_out = self.residual_connections[1](x_out, self.feed_forward)
        return x_out
    
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        x_out = x
        for layer in self.layers:
            x_out = layer(x_out, mask)
        return self.norm(x_out)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_atention: MultiHeadAttention, feed_forward: FeedForward, dropout: float | None):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_atention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        x_out = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x_out = self.residual_connections[1](x_out, lambda x_out: self.cross_attention(x_out, encoder_output, encoder_output, source_mask))
        x_out = self.residual_connections[2](x_out, self.feed_forward)
        return x_out
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        x_out = x
        for layer in self.layers:
            x_out = layer(x_out, encoder_output, source_mask, target_mask)
        return self.norm(x_out)
    
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
    
    
# only encoder - useful for classification task
class TransformerEncoder(nn.Module):
    def __init__(self, encoder: Encoder, embed: InputEmbeddings, pe: PositionEncoding, proj):
        super().__init__()
        self.encoder = encoder
        self.embed = embed
        self.pe = pe
        self.proj = proj
    
    def forward(self, x, mask):
        x_out = self.embed(x)
        x_out = self.pe(x_out)
        return self.proj(self.encoder(x_out, mask))
    

# full transformer
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embed: InputEmbeddings, target_embed: InputEmbeddings, source_pe: PositionEncoding, target_pe: PositionEncoding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pe = source_pe
        self.target_pe = target_pe
        self.proj = proj
        
    def encode(self, x, source_mask):
        x_out = self.source_embed(x)
        x_out = self.source_pe(x_out)
        return self.encoder(x_out, source_mask)
    
    def decode(self, x, encoder_output, source_mask, target_mask):
        x_out = self.target_embed(x)
        x_out = self.target_pe(x_out)
        return self.decoder(x_out, encoder_output, source_mask, target_mask)
    
    def forward(self, source_input, target_input, source_mask, target_mask):
        encoder_output = self.encode(source_input, source_mask)
        decoder_output = self.decode(target_input, encoder_output, source_mask, target_mask)
        return self.proj(decoder_output)
    
    
# building functions
# for encoder-only transformer
def build_encoder_transformer(vocab_size: int, seq_len: int, dropouts: Dict[str, float | None] | None, d_model: int=512, d_ff: int=2048,  N: int=6, h: int=8):
    key_list = ['encoder_pe', 'encoder_self_attetion', 'encoder_feed_forward', 'encoder_block']
    if dropouts is None:
        dropouts = dict()
    for key in key_list:
        if key not in dropouts.keys():
            dropouts[key] = None
        
    embed = InputEmbeddings(d_model, vocab_size)
    pos = PositionEncoding(d_model, seq_len, dropouts['encoder_pe'])
    layers = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropouts['encoder_self_attetion'])
        feed_forward = FeedForward(d_model, d_ff, dropouts['encoder_feed_forward'])
        layers.append(EncoderBlock(self_attention, feed_forward, dropouts['encoder_block']))
    encoder = Encoder(nn.ModuleList(layers))
    proj = nn.Linear(d_model, vocab_size)
    
    transformer_encoder = TransformerEncoder(encoder, embed, pos, proj)
    
    # Xavier uniform distribution for parameter initialization
    for param in transformer_encoder.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
            
    return transformer_encoder
    

# for full transformer
def build_transformer(source_vocab_size: int, target_vocab_size: int, source_seq_len: int, target_seq_len: int, dropouts: Dict[str, float | None] | None, d_model: int=512, d_ff: int=2048,  N: int=6, h: int=8):
    key_list = ['encoder_pe', 'encoder_self_attetion', 'encoder_feed_forward', 'encoder_block', 'decoder_pe', 'decoder_self_attention', 'decoder_cross_attention', 'decoder_feed_forward', 'decoder_block']
    if dropouts is None:
        dropouts = dict()
    for key in key_list:
        if key not in dropouts.keys():
            dropouts[key] = None
        
    # encoder
    source_embed = InputEmbeddings(d_model, source_vocab_size)
    source_pe = PositionEncoding(d_model, source_seq_len, dropouts['encoder_pe'])
    encoder_layers = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropouts['encoder_self_attetion'])
        feed_forward = FeedForward(d_model, d_ff, dropouts['encoder_feed_forward'])
        encoder_layers.append(EncoderBlock(self_attention, feed_forward, dropouts['encoder_block']))
    encoder = Encoder(nn.ModuleList(encoder_layers))
    
    # decoder
    target_embed = InputEmbeddings(d_model, target_vocab_size)
    target_pe = PositionEncoding(d_model, target_seq_len, dropouts['decoder_pe'])
    decoder_layers = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropouts['decoder_self_attention'])
        cross_attention = MultiHeadAttention(d_model, h, dropouts['decoder_cross_attention'])
        feed_forward = FeedForward(d_model, d_ff, dropouts['decoder_feed_forward'])
        decoder_layers.append(DecoderBlock(self_attention, cross_attention, feed_forward, dropouts['decoder_block']))
    decoder = Decoder(nn.ModuleList(decoder_layers))
    
    proj = ProjectionLayer(d_model, target_vocab_size)
    
    transformer = Transformer(encoder, decoder, source_embed, target_embed, source_pe, target_pe, proj)
    
    # Xavier uniform distribution for parameter initialization
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    return transformer
    