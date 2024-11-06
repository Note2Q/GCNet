import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from configure import *


class NCE_C_Parameter(torch.nn.Module):
    def __init__(self, N):
        super(NCE_C_Parameter, self).__init__()
        self.NCE_C = nn.Parameter(torch.zeros(N, requires_grad=True))


class GNN_EBM_Layer_node(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GNN_EBM_Layer_node, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_layer = torch.nn.Linear(input_dim, output_dim)
        self.mlp = torch.nn.Linear(input_dim, output_dim)
        self.args= args

    def node_message_passing(self, x, edge, A_masked):
        T = x.size()[1]
        node_in, node_out = edge[0], edge[1]  
        x_out = torch.index_select(x, 1, node_out) # B, M, d
        x_out = x_out.transpose(1,2)
        x_out = torch.mul(x_out, A_masked)
        x_out = x_out.transpose(1,2) # B, M, d

        update = scatter_add(x_out, node_in, dim=1, dim_size=T)  # B, T, d
        x = x + update  # B, T, d
        return x

    def forward(self, x_1st, x_2nd, edge, A_masked):
        '''
        :param x: (B, T, 2, d)
        :param x_2nd: (B, M, 4, d)
        :param edge: (M, 2)
        :return: (B, T, 2, d_out)
        '''
        node_i_indice = torch.LongTensor([0, 0, 1, 1]).to(x_1st.device)
        node_j_indice = torch.LongTensor([0, 1, 0, 1]).to(x_1st.device)

        x_1st_neg = x_1st[:, :, 0, :]  # B, T, d
        x_1st_pos = x_1st[:, :, 1, :]  # B, T, d

        x_neg = self.node_message_passing(x_1st_neg, edge, A_masked)  # B, T, d
        x_pos = self.node_message_passing(x_1st_pos, edge, A_masked)  # B, T, d
        x = torch.stack([x_neg, x_pos], dim=2)  # B, T, 2, d
        x = self.node_layer(x)  # B, T, 2, d

        return x

class GNN_EBM_Layer_edge(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GNN_EBM_Layer_edge, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_layer = torch.nn.Linear(input_dim, output_dim)
        self.mlp = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x_1st, x_2nd, edge, A_masked):
        '''
        :param x: (B, T, 2, d)
        :param x_2nd: (B, M, 4, d)
        :param edge: (M, 2)
        :return: (B, T, 2, d_out)
        '''
        node_i_indice = torch.LongTensor([0, 0, 1, 1]).to(x_1st.device)
        node_j_indice = torch.LongTensor([0, 1, 0, 1]).to(x_1st.device)
        # print("x_1st{}".format(x_1st.shape))

        edge_i = torch.index_select(x_1st, 1, edge[0])  # B, M, 2, dim
        edge_i = torch.index_select(edge_i, 2, node_i_indice)  # B, M, 4, dim

        edge_j = torch.index_select(x_1st, 1, edge[1])  # B, M, 2, dim
        edge_j = torch.index_select(edge_j, 2, node_j_indice)  # B, M, 4, dim

        edge = x_2nd + self.mlp(edge_i + edge_j) # B, M, 4, d
        edge = self.edge_layer(edge)  # B,M,4,d

        return edge


class GNN_Energy_Model_1st(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, output_dim=1, dropout=0, concat=False):
        super(GNN_Energy_Model_1st, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim  
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1  
        self.dropout = dropout
        self.output_dim = output_dim
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num  

        self.node_hidden_layers = torch.nn.ModuleList()
        self.edge_hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.node_hidden_layers.append(GNN_EBM_Layer_node(in_, out_, args))
            self.edge_hidden_layers.append(GNN_EBM_Layer_edge(in_, out_, args))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim
        self.node_readout = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, output_dim)
        )
        return

    def forward(self, x_1st, x_2nd, edge, A_causal, A_trivial, use_trivial=False):
        '''
        equal 4
        :param x_1st: B,T,2,dim  hi(x)
        :param x_2nd: B,M,4,dim  hij(x)
        :param edge: 2,M
        :return: B,T,1
        '''
        B, T = x_1st.size()[:2]
        h_node_list = [x_1st]
        h_node_list_t = [x_1st]
        x_node, x_edge = x_1st, x_2nd

        if use_trivial:
            for i in range(self.ebm_GNN_layer_num):
                x_node_t = self.node_hidden_layers[i](x_node, x_edge, edge, A_trivial)
                if i < self.ebm_GNN_layer_num - 1:
                    x_node_t = F.relu(x_node_t)
                    # x_edge = F.relu(x_edge)
                x_node_t = F.dropout(x_node_t, self.dropout, training=self.training)
                # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
                h_node_list_t.append(x_node_t)

            if self.concat:
                h_t = torch.cat(h_node_list_t, dim=3).view(B, T, -1)
            else:
                h_t = x_node_t.view(B, T, -1)
            ht = self.node_readout(h_t)
            return ht

        else:
            for i in range(self.ebm_GNN_layer_num):
                x_node = self.node_hidden_layers[i](x_node, x_edge, edge, A_causal)
                if i < self.ebm_GNN_layer_num - 1:
                    x_node = F.relu(x_node)
                    # x_edge = F.relu(x_edge)
                x_node = F.dropout(x_node, self.dropout, training=self.training)
                # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
                h_node_list.append(x_node)

            if self.concat:
                h = torch.cat(h_node_list, dim=3).view(B, T, -1)  # B, T, 2*layer_num*d
            else:
                h = x_node.view(B, T, -1)  # B, T, 2*d
            h = self.node_readout(h)  # B, T, 1
            return h


class GNN_Energy_Model_2nd(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, dropout=0, concat=False):
        super(GNN_Energy_Model_2nd, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1
        self.dropout = dropout
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num

        self.node_hidden_layers = torch.nn.ModuleList()
        self.edge_hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.node_hidden_layers.append(GNN_EBM_Layer_node(in_, out_, args))
            self.edge_hidden_layers.append(GNN_EBM_Layer_edge(in_, out_, args))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim

        self.node_readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, 1)
        )
        self.edge_readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, 1)
        )
        return

    def forward(self, x_1st, x_2nd, edge, A_causal, A_trivial=None, use_trivial=False):
        '''
        :param x_1st: B,T,2,dim
        :param x_2nd: B,M,4,dim
        :param edge: 2,M
        :return: (B,T,2), (B,M,4)
        '''
        # B, T = x_1st.size()[:2]
        B, T, D = x_1st.size()[:3]
        M = edge.size()[1]
        h_node_list = [x_1st]
        h_edge_list = [x_2nd]
        h_node_list_t = [x_1st]
        h_edge_list_t = [x_2nd]
        x_node, x_edge = x_1st, x_2nd
        if use_trivial:
            for i in range(self.ebm_GNN_layer_num):
                x_node_t = self.node_hidden_layers[i](x_node, x_edge, edge, A_trivial)
                x_edge_t = self.edge_hidden_layers[i](x_node, x_edge, edge, A_trivial)
                if i < self.ebm_GNN_layer_num - 1:
                    x_node_t = F.relu(x_node_t)
                    x_edge_t = F.relu(x_edge_t)
                x_node_t = F.dropout(x_node_t, self.dropout, training=self.training)
                x_edge_t = F.dropout(x_edge_t, self.dropout, training=self.training)
                h_node_list_t.append(x_node_t)
                h_edge_list_t.append(x_edge_t)

            if self.concat:
                h_t = torch.cat(h_node_list_t, dim=3)
                h_e_t = torch.cat(h_edge_list_t, dim=3)

            else:
                h_t = h_node_list_t
                h_e_t = h_edge_list_t

            ht = self.node_readout(h_t)
            h_e_t = self.node_readout(h_e_t)
            h_t = ht.squeeze(3)  # B, T, 2
            h_e_t = h_e_t.squeeze(3)  # B, M, 4
            return h_t, h_e_t
        else:
            for i in range(self.ebm_GNN_layer_num):
                x_node = self.node_hidden_layers[i](x_node, x_edge, edge, A_causal)
                x_edge = self.edge_hidden_layers[i](x_node, x_edge, edge, A_causal)
                if i < self.ebm_GNN_layer_num - 1:
                    x_node = F.relu(x_node)
                    x_edge = F.relu(x_edge)
                x_node = F.dropout(x_node, self.dropout, training=self.training)
                x_edge = F.dropout(x_edge, self.dropout, training=self.training)
                h_node_list.append(x_node)
                h_edge_list.append(x_edge)

            if self.concat:
                h_node = torch.cat(h_node_list, dim=3)  
                h_edge = torch.cat(h_edge_list, dim=3)  
            else:
                h_node = x_node  # B, T, 2, d
                h_edge = x_edge  # B, M, 4, d
            h_node = self.node_readout(h_node)  # B, T, 2, 1
            h_edge = self.edge_readout(h_edge)  # B, M, 4, 1
            h_node = h_node.squeeze(3)  # B, T, 2
            h_edge = h_edge.squeeze(3)  # B, M, 4
            return h_node, h_edge
