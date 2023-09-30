import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class transMM_FC (nn.Module):
    def __init__(self, time_steps=64, out_feats=104, criterion='MSELoss'):
        super(transMM_FC, self).__init__()
        self.orig_d_l = 300
        self.orig_d_a = 128
        self.d_l = 60
        self.d_a = 60
        self.aonly = True
        self.lonly = False
        self.num_heads = 5
        self.layers = 6
        self.relu_dropout = 0.1
        #self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.25
        self.decoder_layer = self.layers
        self.combined_dim = self.d_l + self.d_a

        output_dim = out_feats       
        self.time_steps = time_steps

        #self.criterion = eval('torch.nn.' + criterion)()
    
        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 2. Transformer
        self.trans_self_a = self.get_trans_encoder(self.d_a, self.layers)
        self.trans_self_l = self.get_trans_encoder(self.d_l, self.layers)
        
        # 3. Projection layers 
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        #self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, output_dim)
        self.m = nn.LeakyReLU(negative_slope=0.01)
        #self.out_layer = nn.Linear(self.combined_dim, output_dim)
        self.out_layer = nn.Linear(output_dim, output_dim)

    def get_trans_encoder(self, dim, layers):
        print("embed_dim in encoder: {0}".format(dim))
        encoder_layer = nn.TransformerEncoderLayer(dim, self.num_heads)
        return nn.TransformerEncoder(encoder_layer, layers)
    
    def forward(self, x_a, x_l):
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        print("audio {0} text {1}".format(x_a.shape, x_l.shape))  #batch_size, seq_len, n_features

        # Project the textual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        print("encoder input shape x_a: {0}".format(proj_x_a.shape))

        ha_las = self.trans_self_a(proj_x_a)
        hl_las = self.trans_self_l(proj_x_l)
        #print("ha_las {0} hl_las {1}".format(ha_las.shape, hl_las.shape))
        h_las = torch.cat((ha_las, hl_las), dim=2) #[framce_num, batch_num, dim]
        #print("mm input h_las {0}".format(h_las.shape))

        #decoding part
        internal_losses = []
        #context_conv = conv1(target_context)
        output = self.proj2(F.dropout(self.m(self.proj1(h_las)), p=self.out_dropout, training=self.training))
        output = self.out_layer(output)
        output = output.transpose(0,1)

        #loss = self.criterion(output, eval_attr)
        #internal_losses.append(loss)
        internal_losses = []
        return output, internal_losses
        


