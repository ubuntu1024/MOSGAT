""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from SelfAttention import ScaledDotProductAttention
from torch_geometric.nn import GATConv

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class GAT(torch.nn.Module):
    def __init__(self, in_feats, hgcn_dim,dropout):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hgcn_dim[0], heads=1,concat=False)
        self.conv2 = GATConv(hgcn_dim[0], hgcn_dim[1], heads=1,concat=False)
        self.conv3 = GATConv(hgcn_dim[1], hgcn_dim[2], heads=1,concat=False)
        self.dropout = dropout

    def forward(self, data,adj):
        x= data
        edge_index =adj
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        return x




class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class TCP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.selfattentionLayer=Self_attention(hidden_dim[0])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, feature, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        TCPLogit, TCPConfidence =   dict(), dict()
        all_cord = []
        for view in range(self.views):
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]
            all_cord.append(feature[view])
        MMfeature = self.selfattentionLayer(all_cord)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class Self_attention(nn.Module):
    def __init__(self,dim_he_list):
        super().__init__()
        self.ScaledDotProductAttention=ScaledDotProductAttention(d_model=dim_he_list, d_k=dim_he_list, d_v=dim_he_list, h=6)
        self.ScaledDotProductAttention.apply(xavier_init)
        self.dropout=0.5

    def forward(self, in_list):
        num_view = len(in_list)
        num_sam = in_list[0].shape[0]
        num_feat = in_list[0].shape[1]
        if num_view ==3:
            out_feat =torch.stack([in_list[0], in_list[1],in_list[2]], dim=1)
        if num_view ==2:
            out_feat =torch.stack([in_list[0], in_list[1]], dim=1)
        if num_view == 1:
            out_feat =in_list[0]
        SA_out_feat=self.ScaledDotProductAttention(out_feat,out_feat,out_feat)
        SA_out_feat = F.leaky_relu(SA_out_feat, 0.25)
        SA_out_feat = F.dropout(SA_out_feat, self.dropout, training=self.training)
        SA_out_feat=SA_out_feat.reshape(-1,num_view*num_feat)
        return SA_out_feat



def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, model_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GAT(dim_list[i], dim_he_list, model_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    model_dict["Fus"]=TCP(dim_list,[dim_he_list[-1]],num_class,model_dopout)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()),
                lr=lr_e)
    optim_dict["M"] = torch.optim.Adam(list(model_dict["Fus"].parameters()),lr=lr_e)

    return optim_dict