import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from gnn_tox.gnn.gat import GATConv
from gnn_tox.gnn.gcn import GCNConv
from gnn_tox.gnn.graphsage import GraphSAGEConv
from gnn_tox.gnn.gin import GINConv

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 4
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK='last', drop_ratio=0, gnn_type='gin'):
        super(GNN, self).__init__()
        """ init func
        Args:
            num_layer (int): the number of GNN layers
            emb_dim (int): dimensionality of embeddings
            JK (str): last, concat, max or sum.
            drop_ratio (float): dropout rate
            gnn_type (str): gin, gcn, graphsage, gat

        """
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == 'gin':
                self.gnns.append(GINConv(emb_dim, aggr = 'add'))
            elif gnn_type == 'gcn':
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == 'gat':
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == 'graphsage':
                self.gnns.append(GraphSAGEConv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError('unmatched number of arguments.')
        center_atom_props = x[:, 2]
        center_atom_indices = (center_atom_props == 1).nonzero(as_tuple=True)[0]
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == 'concat':
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == 'max':
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == 'sum':
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation, center_atom_indices


def mean_and_sum(x, batch):
    mean = global_mean_pool(x, batch)
    sum = global_add_pool(x, batch)
    return torch.cat((mean, sum), 1)


class GNN_graphpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, num_tasks, JK = 'last', drop_ratio = 0, graph_pooling = 'mean', gnn_type = 'gin'):
        """ init func

        Args:
            num_layer (int): the number of GNN layers
            emb_dim (int): dimensionality of embeddings
            num_tasks (int): number of tasks in multi-task learning scenario
            JK (str): last, concat, max or sum
            drop_ratio (float): dropout rate
            graph_pooling (str): sum, mean, max, attention, set2set
            gnn_type: gin, gcn, graphsage, gat

        """
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kinds of graph pooling
        if graph_pooling == 'sum':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'mean_sum':
            self.pool = mean_and_sum
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == 'attention':
            if self.JK == 'concat':
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == 'set2set':
            set2set_iter = int(graph_pooling[-1])
            if self.JK == 'concat':
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError('Invalid graph pooling type.')

        # For graph-level binary classification
        if graph_pooling[:-1] == 'set2set':
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == 'concat':
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Sequential(
                nn.Dropout(0.05),
                nn.Linear(self.mult * self.emb_dim * 2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, self.num_tasks)
        )

    def forward(self, *argv):       
        x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        node_representation, center_atom_indices = self.gnn(x, edge_index, edge_attr)
        node_representation = torch.index_select(node_representation, 0, center_atom_indices)
        batch = torch.index_select(batch, 0, center_atom_indices)
        return self.graph_pred_linear(self.pool(node_representation, batch))



