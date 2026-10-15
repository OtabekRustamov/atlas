import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2,hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, MAX_LEN=1000):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # Transformer on protein sequence
        # self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        # self.pos_embed = PositionalEncoding(embed_dim, MAX_LEN, device)
        self.embed_dim = embed_dim
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)  ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))  ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, MAX_LEN + 1, embed_dim))  ## position embedding

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, mask1):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # 1D Conv
        embedded_xt = self.embed(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = conv_xt.view(-1, 32 * 121)

        # flatten
        conv_xt = conv_xt.view(-1, 32 * 121)
        conv_xt = self.fc1_xt(conv_xt)

        # Transformer
        # emb_xt = self.embed(target)
        # emb_xt = self.dropout(emb_xt + self.pos_embed(target))

        xt = self.embed(target) * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(xt.shape[0], -1, -1)
        xt = torch.cat((cls_token, xt), dim=1)

        xt = xt + self.pos_embed
        xt = xt.permute(1, 0, 2)
        xt = self.transformer_encoder(xt, src_key_padding_mask=mask1)
        xt = xt.permute(1, 0, 2)
        xt = self.norm(xt)
        xt = xt[:, 0]

        xt = conv_xt + xt

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
