import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, device, heads = 3):
        super(SelfAttention, self).__init__()
        self.heads = heads

        self.attention_linear = nn.Linear(embed_dim*heads, embed_dim, bias=False)
    
        self.keys = [nn.Linear(embed_dim, embed_dim, bias = False).to(device) for _ in range(heads)]
        self.values = [nn.Linear(embed_dim, embed_dim, bias = False).to(device) for _ in range(heads)]
        self.queries = [nn.Linear(embed_dim, embed_dim, bias = False).to(device) for _ in range(heads)]

    def forward(self, values, keys, queries, mask = None):
        keys = [self.keys[i](keys) for i in range(self.heads)]
        queries = [self.queries[i](queries) for i in range(self.heads)]
        values = [self.values[i](values) for i in range(self.heads)]

        temp = [torch.einsum("bwe,bne->bwn",[queries[i], keys[i]]) / keys[i].shape[2] for i in range(self.heads)]   #(Batch size, Number of words, Number of words)
        if mask is not None:
            temp = [temp[i].masked_fill(mask == 0, -1e20) for i in range(self.heads)]

        attentions = [torch.softmax(temp[i], dim = 2) @ values[i] for i in range(self.heads)]   #(Batch size, Number of words, embed size)
        attention = torch.concat([attentions[i] for i in range(self.heads)], dim = 2)  #(Batch size, Number of words, embed size * heads)
        attention = self.attention_linear(attention)    #(Batch size, Number of words, embed size)

        return attention
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, ff_expansion, num_heads, device):
        super(Encoder, self).__init__()
        self.self_attention = SelfAttention(embed_dim, heads = num_heads, device = device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_expansion*embed_dim), nn.ReLU(),
            nn.Linear(embed_dim*ff_expansion, embed_dim)
        )
    
    def forward(self, x):
        A = self.self_attention(x,x,x)
        x = self.norm1(x + A)
        ff = self.ff(x)
        x = self.norm2(ff + x)  #Same as input dimension
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, ff_expansion, num_heads, device):
        super(Decoder, self).__init__()
        self.device = device
        self.masked_attention = SelfAttention(embed_dim = 1, heads = num_heads, device = device)
        self.attention = SelfAttention(embed_dim, heads=num_heads, device = device)
        
        self.norm1 = nn.LayerNorm(1)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(1, embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_expansion*embed_dim), nn.ReLU(),
            nn.Linear(embed_dim*ff_expansion, embed_dim)
        )

    def get_mask(self, target):
        n = target.shape[1]
        mask = torch.ones(n,n).tril()
        mask.to(self.device)
        return 
        
    def forward(self, K, V, trg):
        mask = self.get_mask(trg)
        Q = self.masked_attention(trg, trg, trg, mask)
        Q = self.norm1(Q + trg)
        Q = self.linear(Q)

        x = self.attention(V, K, Q, mask)
        x = self.norm2(x + Q)
        return self.norm3(self.ff(x) + x)   #Same as input dimension

class Transformer(nn.Module):
    def __init__(self, embed_dim, 
                 enc_expansion, dec_expansion,
                 enc_heads, dec_heads,
                 num_enc, num_dec, device, 
                 pos_enc = True):
        super(Transformer, self).__init__()
        self.pos_enc_bool = pos_enc
        self.device = device

        self.encoder = nn.ModuleList([Encoder(embed_dim=embed_dim, ff_expansion=enc_expansion, num_heads=enc_heads, device = device)
                                      for _ in range(num_enc)])
        self.decoder = nn.ModuleList([Decoder(embed_dim=embed_dim, ff_expansion=dec_expansion, num_heads=dec_heads, device = device) 
                                      for _ in range(num_dec)])
        
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def position_encoding(self, x):
        d = x.shape[2]
        n = x.shape[1]
        pos_enc = torch.zeros(n,d)

        i = 2*torch.arange(0,d)/d
        pos = torch.arange(0,n).unsqueeze(1)

        pos_enc[:, 0::2] = torch.sin(pos/(1000**i))[:,::2]
        pos_enc[:, 1::2] = torch.cos(pos/(1000**i))[:,1::2]
        pos_enc = pos_enc.to(self.device)

        return pos_enc  #same dimensions as x
        
    def forward(self, x, y):
        #Adding positional encodings
        if self.pos_enc_bool:
            x = x + self.position_encoding(x)
            y = y + self.position_encoding(y)

        #Encoding Layer
        for layer in self.encoder:
            x = layer(x)
            
        #Decoding Layer
        for layer in self.decoder:
            x = layer(x,x,y)

        return self.output_layer(x)

class Data(Dataset):
    def __init__(self, data : torch.Tensor, win_len, stride, transform = None):
        print(data.shape)
        self.len = data.shape[0]
        self.features = data[:,:-1]
        self.labels = data[:,-1].unsqueeze(1)

        self.transform = transform
        self.win_len = win_len
        self.stride = stride
        self.num_wins = self.floor(((data.shape[0] - win_len) / stride) + 1)

    def floor(self, num):
        num = torch.tensor(num)
        return torch.floor(num)

    def __getitem__(self, index):
        assert(index <= self.num_wins), f"Index ({index}) out of bouund, max index is {int(self.num_wins)}"
        if self.transform is not None:
            self.features[index : index + self.win_len, :] = self.transform(self.features[index : index + self.win_len, :])
        
        return (self.features[index : index + self.win_len, :],
                self.labels[index : index + self.win_len, :],)
    
    def __len__(self):
        return int(self.num_wins)

class MeanNormalization():
    def __init__(self):
        pass

    def __call__(self, features):
        m = torch.mean(features, dim = 0)
        std = torch.std(features, dim = 0)
        return ((features - m) / (std + 1e-10))

def get_size(model):
    size_model = sum(
        param.numel() * torch.finfo(param.data.dtype).bits
        if param.data.is_floating_point()
        else param.numel() * torch.iinfo(param.data.dtype).bits
        for param in model.parameters())
    return f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB"
