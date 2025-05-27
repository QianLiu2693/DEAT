import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.linalg import eig

from pickle import NONE
import numpy as np
import numba as nb
import scipy.sparse as sp
import scipy
import os
from multiprocessing import Pool
from log import logger

class SimpleGNN(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 hidden_size,
                 dropout_rate,
                 depth,
                 device):
        super(SimpleGNN, self).__init__()  
        self.node_size = node_size
        self.rel_size = rel_size
        self.hidden_size = hidden_size
        self.dropout = dropout_rate
        self.depth = depth
        self.device = device

        self.ent_emb = nn.Embedding(node_size,hidden_size)
        self.rel_emb = nn.Embedding(rel_size,hidden_size)
        self.adj_emb = nn.Embedding(node_size,hidden_size)
        
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.adj_emb.weight)
        

    def forward(self, data):
        adj = data[0]
        ent_ent = data[1]
        ent_rel = data[2]
        time_dict = data[3]
       
        adj = adj.to(device=self.device)
        ent_ent = ent_ent.to(device=self.device)
        ent_rel = ent_rel.to(device=self.device)
        keys_tensor =F.one_hot( torch.tensor([key for key in time_dict.keys()])).float().to(device=self.device)
        #print('keys_tensor',keys_tensor.shape)

        
        he_emb = self.ent_emb.weight
        hr_emb = self.rel_emb.weight 
        adj_emb = self.adj_emb.weight 
        
        ent_ent = torch.mm(ent_ent,keys_tensor)
        adj=torch.mm(adj,keys_tensor)
        #torch.mm(a, b) 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。
        #ent_rel =torch.mm(ent_ent,ent_rel)
        
        he = torch.matmul(ent_ent, he_emb)
        hr = torch.matmul(ent_rel, hr_emb)
        hadj = torch.matmul(adj, adj_emb)
        
        #################################
        adj=F.leaky_relu(hadj)
        adj1=self.modify_svd(adj)
        he = torch.einsum('ik,kj->ij', [he, adj1])
        hr = torch.einsum('ik,kj->ij', [hr, adj1])

        #################################
        
        adj = self.modify_attention(self.modify_svd_U(adj))

        
        h = torch.cat([he,hr,adj],-1)
        
        h = torch.tanh(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        hg = h 
        for i in range(self.depth-1):
            h = torch.matmul(ent_ent, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)        
            hg = torch.cat([hg,h],-1) 
            hg = self.modify_attention(hg) 

        h_mul = hg 
        return h_mul
        
    def modify(self, data):
         number = data.shape[1]
         #data = self.modify_transformer(data)
         data = self.modify_gru(data)
         data = self.modify_mlp(data)
         data = self.modify_attention(data) 
         return data
         
    def modify_gru(self, data):
         number = data.shape[1]
         gruCell = nn.GRUCell(number, number).to(self.device)
         data = gruCell(data).to(self.device)
         return data
    def normalize_adj(adj):
       adj = sp.coo_matrix(adj)
       rowsum = np.array(adj.sum(1))
       d_inv_sqrt = np.power(rowsum, -0.5).flatten()
       d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
       d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
       return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T
    def modify_gruA(self, data):
         number = data.shape[1]
         gruA = GRUA(number, number,data.shape[0],0.1).to(self.device)
         data =data.unsqueeze(0)
         data = gruA(data,data).to(self.device)
         return data
    def modify_mlp(self, data):
         number = data.shape[1]
         mlp_ratio = 4        
         mlp = MLP(number, number * mlp_ratio, dropout = 0.2).to(self.device)
         data = mlp(data).to(self.device)
         #hg = torch.einsum('ij,ij->ij', [hg,  self.modify_mlp(hg)]) 
         return data
    def modify_attention(self, data):
         number = data.shape[1]
         ################增加
         selfAttention =SelfAttention(number).to(self.device)
         h_middle = selfAttention(data.view(data.shape[0],1,-1)).to(self.device)
         data = h_middle.view(-1,number).to(self.device) 
         return data
    def modify_transformer(self,data):
         number = data.shape[1]
         transformer = Transformer(number, number, number, 8, 3, 0.1).to(self.device)
         data = transformer(data.view(data.shape[0],1,-1)).view(-1,number).to(self.device)
         return data
    def modify_coAttention(self,data):
         number = data.shape[1]
         coAttention=CoAttention(number).to(self.device)
         h_middle = coAttention(data.view(data.shape[0],1,-1)).to(self.device)
         data = h_middle.view(-1,number).to(self.device) 
         return data
    def modify_DRNN(self,data):
         data = data.unsqueeze(2)
         number0=data.shape[0]
         number1 = data.shape[1]
         number2 = data.shape[2]
         drnn=DRNN(number2,number0,number1).to(self.device)
         h_middle = drnn(data).to(self.device)
         data = h_middle.view(-1,number2).to(self.device) 
         return data
    def modify_simplelstm(self,data):
         number = data.shape[1]
         lstm_layer = nn.LSTM(input_size=number, hidden_size=number, num_layers=2).to(self.device)
         out2, (h_n, c_n) = lstm_layer(data.view(-1,1,data.shape[1]))
         data = out2.view(-1,number)
         return data
    def modify_lstms(self,data):
        number = data.shape[1]
        lstm_layer = nn.LSTM(input_size=number, hidden_size=number, num_layers=2).to(self.device)
        out1, (h_n, c_n) = lstm_layer(data.view(-1,1,data.shape[1]))
        data = out1.view(-1,number)

        data=F.softmax(data)
        return data
    def modify_eighnvalue(self,data):
    #计算特征值和特征向量 data必须是方阵
         eigenvalues, eigenvectors = eig(data)
         print('data@@@@@@@@@',data.shape)
         print('eigenvalues@@@@@@@@@',eigenvalues.shape)
         print('eigenvectors@@@@@@@@@',eigenvectors.shape)
         return data
    def modify_svd(self,data):
    #可以使用torch.svd函数对矩阵进行秩分解，该函数返回奇异值分解的结果，其中包括U、S、V三个矩阵。其中，U和V是正交矩阵，S是对角矩阵，其中的元素是矩阵A的奇异值。
         U, S, V = torch.svd(data)
         #print('data@@@@@@@@@',data.shape)
         #print('U@@@@@@@@@',U.shape)
         #print('S@@@@@@@@@',S.shape)
         #print('V@@@@@@@@@',V.shape)         
         return V
    def modify_svd_U(self,data):
    #可以使用torch.svd函数对矩阵进行秩分解，该函数返回奇异值分解的结果，其中包括U、S、V三个矩阵。其中，U和V是正交矩阵，S是对角矩阵，其中的元素是矩阵A的奇异值。
         U, S, V = torch.svd(data)
         #print('data@@@@@@@@@',data.shape)
         #print('U@@@@@@@@@',U.shape)
         #print('S@@@@@@@@@',S.shape)
         #print('V@@@@@@@@@',V.shape)         
         return U
         
    def modify_cov(self,data):
         #print('data@@@@@@@@@',data.shape)
         conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1).to(self.device)
         # 定义一个矩阵
         matrix = data.view(1, 1, data.shape[0], data.shape[1]).to(self.device)
         # 通过卷积层处理矩阵
         output = conv(matrix).view(-1,data.shape[1]).to(self.device)
          # 对输出矩阵进行归一化，使其值在0-1之间
         output = torch.nn.functional.normalize(output, p=2, dim=1)
         #print('output@@@@@@@@@',output.shape)
         return output
    def add_feature(self,emb_time_1,emb_time_2,x_input):
        num = x_input.shape[1]
        if emb_time_1.shape[0]<num:
             number = int(num//emb_time_1.shape[0])+1
             emb_time_1=emb_time_1.repeat(number,1)
             emb_time_2=emb_time_2.repeat(number,1)
        #print('**time***',emb_time_1.shape)
        new_size = ( num,  num)
        emb_time_1 = torch.flatten(emb_time_1)  # 先将二维tensor转为一维tensor
        emb_time_1 = emb_time_1[:new_size[0] * new_size[1]]  # 截取前new_size[0] * new_size[1]个元素
        emb_time_1 = emb_time_1.reshape(new_size)  # 将一维tensor转为指定维度的二维tensor
        emb_time_2 = torch.flatten(emb_time_2)  # 先将二维tensor转为一维tensor
        emb_time_2 = emb_time_2[:new_size[0] * new_size[1]]  # 截取前new_size[0] * new_size[1]个元素
        emb_time_2 = emb_time_2.reshape(new_size)  # 将一维tensor转为指定维度的二维tensor
        x_input=torch.einsum('ik,kj->ij', [x_input, emb_time_1])
        x_input=torch.einsum('ik,kj->ij', [x_input, emb_time_2])
        return x_input
    
                  
class Alignment_loss(nn.Module):
    def __init__(self,
                 gamma,
                 batch_size,
                 device
                 ):

        super(Alignment_loss, self).__init__()

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

    def forward(self, outfeature, trainset):
        h = outfeature.to(device=self.device)
        set = torch.as_tensor(trainset).to(device=self.device)
        def _cosine(x):
            dot1 = torch.matmul(x[0], x[1], axes=1)
            dot2 = torch.matmul(x[0], x[0], axes=1)
            dot3 = torch.matmul(x[1], x[1], axes=1)
            #print('dot1',dot1.shape)
            #print('dot2',dot2.shape)
            #print('dot3',dot3.shape)
            max_ = torch.maximum(torch.sqrt(dot2 * dot3), torch.epsilon())
            #print('max_',max_.shape)
            return dot1 / max_
    
        def l1(ll,rr):
            return torch.sum(torch.abs(ll-rr),axis=-1,keepdims=True)
    
        def l2(ll,rr):
            return torch.sum(torch.square(ll-rr),axis=-1,keepdims=True)
        
        l,r,fl,fr = [h[set[:,0]],h[set[:,1]],h[set[:,2]],h[set[:,3]]]
        loss = F.relu(self.gamma + l1(l,r) - l1(l,fr)) + F.relu(self.gamma + l1(l,r) - l1(fl,r))
        loss_avg = torch.sum(loss,0,True) / self.batch_size
        return loss_avg
    
#########################################################################################################
class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()  
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)
##########################################################################################################
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers = 8, kernel_size = 5, mask_next = True, mask_diag = False):
        super().__init__()
        
        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag
        
        h = headers
        
        padding = (kernel_size-1)
        self.padding_opertor = nn.ConstantPad1d((padding,0), 0)
        
        self.toqueries = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tokeys = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tovalues = nn.Conv1d(k, k*h, kernel_size = 1 , padding=0 ,bias=False) 
        
        self.unifyheads = nn.Linear(k*h, k)
    def forward(self, x):
        b, t, k  = x.size()
        
        assert self.k == k, 'Number of time series '+str(k)+' didn t much the number of k '+str(self.k)+' in the initiaalization of the attention layer.'
        h = self.headers
        x = x.transpose(1,2)
        x_padded = self.padding_opertor(x)
        queries = self.toqueries(x_padded).view(b,k,h,t)
        keys = self.tokeys(x_padded).view(b,k,h,t)
        values = self.tovalues(x).view(b,k,h,t)
        queries = queries.transpose(1,2)
        queries = queries.transpose(2,3)
        
        values = values.transpose(1,2)
        values = values.transpose(2,3) 
        
        keys = keys.transpose(1,2)
        keys = keys.transpose(2,3)
        
        queries = queries/(k**(.25))
        keys = keys/(k**(.25))
        
        queries = queries.transpose(1,2).contiguous().view(b*h, t, k)
        keys = keys.transpose(1,2).contiguous().view(b*h, t, k)
        values = values.transpose(1,2).contiguous().view(b*h, t, k)
        
        weights = torch.bmm(queries, keys.transpose(1,2))
        if self.mask_next :
            if self.mask_diag :
                indices = torch.triu_indices(t ,t , offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
            else :
                indices = torch.triu_indices(t ,t , offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')
        weights = F.softmax(weights, dim=2)
        output = torch.bmm(weights, values)
        output = output.view(b,h,t,k)
        output = output.transpose(1,2).contiguous().view(b,t, k*h)
        
        return self.unifyheads(output)  
        
class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, inputs):
        # [batch_size, inputs, embedding_size]
        K = self.query(inputs)
        V = self.query(inputs)
        Q = self.query(inputs)

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0, 2, 1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output

##################################################################################################### 
# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, nheads, nlayers, dropout):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(input_dim, nheads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = torch.transpose(src, 0, 1)
        output = self.transformer_encoder(src)
        output = torch.transpose(output, 0, 1)
        output = self.decoder(output)
        return output
######################################################################################################3
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru_forward = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru_backward = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # 前向传播
        x_forward, _ = self.gru_forward(x)

        # 反向传播
        x_backward, _ = self.gru_backward(torch.flip(x, [1]))

        # 将前向传播和反向传播的结果进行拼接
        #x = torch.cat((x_forward, torch.flip(x_backward, [1])), dim=2)
        x = x_forward +torch.flip(x_backward, [1])
        return x
################自编码
class Autoencoder(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super().__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
        )
        # 定义解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_shape),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 编码输入
        x = self.encoder(x)
        # 解码输出
        x = self.decoder(x)
        return x
#####################
class VarGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VarGRU, self).__init__()

        self.hidden_size = hidden_size

        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.candidate_hidden_state_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)

        update = self.update_gate(combined)
        reset = self.reset_gate(combined)

        reset_hidden = reset * hidden

        combined = torch.cat((input, reset_hidden), dim=1)
        candidate = self.candidate_hidden_state_gate(combined)

        # 更新隐藏状态
        output = (1 - update) * hidden + update * candidate

        return output
###################超图时间复杂度太高####################
class Hypergraph:
    def __init__(self, nodes, hyperedges):
        self.nodes = nodes
        self.hyperedges = hyperedges

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def neighbors(self, node):
        for hyperedge in self.hyperedges:
            if node in hyperedge:
                yield hyperedge

    def adjacency_matrix(self):
        num_nodes = len(self.nodes)
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        for hyperedge in self.hyperedges:
            for node in hyperedge:
                for neighbor in self.neighbors(node):
                    if torch.any(neighbor != hyperedge):
                        node = node.long()  # 如果 node 不是 long tensor
                        neighbor = neighbor.long()  # 如果 neighbor 不是 long tensor
                        adj_matrix[node, neighbor] = 1
        return adj_matrix
##########共注意力机制（Co-Attention Mechanism）：用于优化模型在多模态数据（如图像和文本）处理中的性能。
class CoAttention(nn.Module):
    def __init__(self, input_dim):
        super(CoAttention, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)
        self.output_dim = input_dim

    def forward(self, inputs):
        # inputs: (batch_size, num_modalities, input_dim)
        # 对每个模态进行自注意力计算
        attn = self.linear(inputs)
        attn = self.softmax(attn)
        # 计算共注意力权重
        co_attn = torch.mean(inputs * attn, dim=1)
        # 对共注意力权重进行归一化处理
        co_attn = co_attn / torch.norm(co_attn, p=2, dim=1, keepdim=True)
        # 将共注意力权重应用于输入矩阵
        output = inputs * co_attn[:, None, :]
        return output
class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(DRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出特征图进行全连接层操作
        out = self.fc(out[:, -1, :])
        return out
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        lstm_out, hidden = self.lstm(inputs.view(1, 1, -1), hidden)
        output = self.linear(lstm_out.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))
class GRUA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUA, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        attn_weights = F.softmax(self.attn(out).squeeze(2), dim=1)
        context = torch.mul(out, attn_weights.unsqueeze(2).expand_as(out))
        context = context.sum(dim=1)
        context = self.dropout(context)
        out = self.fc(context)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device)
        return hidden
        
        

