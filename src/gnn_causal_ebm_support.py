import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean

from configure import *
from gnn_casual_ebm import *

bce_criterion = nn.BCEWithLogitsLoss()
ce_criterion = nn.CrossEntropyLoss()
softmax_opt = nn.Softmax(-1)
EPS = 1e-8

class DAG_model(torch.nn.Module):
    def __init__(self, input_dim=2,adj=None,args=args,bias=False):
        super(DAG_model, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ELU(),
            nn.Linear(4, input_dim)
        )
        self.adj = nn.Parameter(adj)
        self.exogenous = nn.Parameter(torch.nn.init.normal_(torch.zeros(self.adj.size()[0]), mean=0, std=1))


    def forward(self, x_mix):
        '''
        :param x: (B, T, 2)
        :param x_2nd: (B, M, 4)
        :param edge: (M, 2)
        :return: (B, T, 1)
        '''

        fi_z = self.net(torch.matmul(self.adj.t(),x_mix)) 
        exo = torch.stack((self.exogenous,self.exogenous),dim=1)
        fi_z = fi_z + exo
        fi_z = softmax_opt(fi_z)
        return fi_z


def matrix_poly(matrix, d, args):
    x = torch.eye(d).to(args.device) + torch.div(matrix.to(args.device), d).to(args.device)
    return torch.matrix_power(x, d)


def _h_A(A, n, args):
    expm_A = matrix_poly(A * A, n, args)
    h_A = torch.trace(expm_A) - n
    return h_A

class mask_A(torch.nn.Module):
    def __init__(self,task_ebm_dim,args):
        super(mask_A, self).__init__()
        self.with_edge_attention = args.with_edge_attention
        self.edge_att_mlp = nn.Linear(task_ebm_dim, 2)
        #self.node_att_mlp = nn.Linear(task_ebm_dim, 2)

    def forward(self,task_repr,task_edge):
        row,col = task_edge

        task_repr = torch.add(task_repr[row], task_repr[col])
        if self.with_edge_attention:
            edge_att = F.softmax(self.edge_att_mlp(task_repr), dim = -1)
        A_causual = edge_att[:,0] 
        A_trivial = edge_att[:,1]


        return A_causual, A_trivial
# ----------------------------------------------------------------------------------------------------------------------
#
# Prediction Function
#
# ----------------------------------------------------------------------------------------------------------------------

# get_GNN_prediction_second_order
def get_GNN_prediction_1st_order_prediction(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model,
        graph_repr, task_repr, task_edge, mask,use_trivial, args):
    B = len(graph_repr)
    T = task_repr.size()[0]
    M = task_edge.size()[1]

    A_causual, A_trivial = mask(task_repr=task_repr, task_edge=task_edge)

    ########## Get 1st-order state prediction ##########
    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order = task_repr.unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st_order = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order)  # B, T, 2*d_ebm
    #first order_prediction_model = MoleculeTaskPredictionModel
    y_pred_1st_order = y_pred_1st_order.view(B, T, 2, -1)  # B, T, 2, d_ebm


    ########## Get 2nd-order state prediction ##########
    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol


    task_repr_2nd_order_node1, task_repr_2nd_order_node2 = mapping_task_repr(task_repr, task_edge, B)
    y_pred_2nd_order = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1, task_repr_2nd_order_node2)  # B, M, 4*d_ebm
    y_pred_2nd_order = y_pred_2nd_order.view(B, M, 4, -1)  # B, M, 4, d_ebm

    y_pred = GNN_energy_model(y_pred_1st_order, y_pred_2nd_order, task_edge, A_causual, A_trivial)  # B, T, 1
    if use_trivial:
        y_trivial_pred  = GNN_energy_model(y_pred_1st_order, y_pred_2nd_order, task_edge, A_causual, A_trivial,use_trivial)
        return y_pred,y_trivial_pred,A_causual
    return y_pred,A_causual


# get_GNN_prediction_second_order_2nd_Order_Prediction
def get_GNN_prediction_2nd_order_prediction(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model,
        graph_repr, task_repr, task_edge, A_causual,adj, mask,use_trivial, args):
    B = len(graph_repr)
    T = task_repr.size()[0]
    M = task_edge.size()[1]

    new_a_causual = torch.zeros_like(A_causual).to(args.device)
    index = 0

    for i in range(M):
        node1 = task_edge[0][i]
        node2 = task_edge[1][i]
        new_a_causual[index] = adj[node1][node2]
        index += 1

    A_causual = new_a_causual


    ########## Get 1st-order prediction ##########
    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order = task_repr.unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order)  # B, T, 2*d_ebm
    y_pred_1st = y_pred_1st.view(B, T, 2, -1)  # B, T, 2, d_ebm

    ########## Get 2nd-order prediction ##########
    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol
    task_repr_2nd_order_node1, task_repr_2nd_order_node2 = mapping_task_repr(task_repr, task_edge, B)  # (B, M, d_task), (B, M, d_task)
    y_pred_2nd = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1, task_repr_2nd_order_node2)  # B, M, 4*d_ebm
    y_pred_2nd = y_pred_2nd.view(B, M, 4, -1)  # B, M, 4, d_ebm

    y_pred_1st_order, y_pred_2nd_order = GNN_energy_model(y_pred_1st, y_pred_2nd, task_edge,  A_causual, use_trivial=False)

    if args.softmax_energy:
        y_pred_1st_order = softmax_opt(y_pred_1st_order)
        if torch.isnan(y_pred_1st_order).any():
            import pdb; pdb.set_trace()
        y_pred_2nd_order = softmax_opt(y_pred_2nd_order)
        if torch.isnan(y_pred_2nd_order).any():
            import pdb; pdb.set_trace()

    return y_pred_1st_order, y_pred_2nd_order





# ----------------------------------------------------------------------------------------------------------------------
#
# Energy Function
#
# ----------------------------------------------------------------------------------------------------------------------

def energy_function_feature(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge, mask, use_trivial, args, **kwargs):

    if args.use_trivial:
        y_pred,y_trivial_pred,A_causual = prediction_function( 
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,use_trivial=use_trivial,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, mask=mask,
        args=args) # B, T, 1
        y_true = y_true.unsqueeze(2)  # B, T, 1
        y_valid = y_true ** 2 > 0
        y_true = ((1 + y_true) / 2)  # B, T, 1

        masked_y_true = torch.masked_select(y_true, y_valid)
        masked_y_pred = torch.masked_select(y_pred, y_valid)
        y_trivial_true = torch.ones_like(y_true)*0.5
        masked_y_trivial_true = torch.masked_select(y_trivial_true, y_valid)
        masked_y_trivial_pred = torch.masked_select(y_trivial_pred, y_valid)
        M = task_edge.size()[1]
        matrix_loss = torch.sum(torch.abs(A_causual) / M)
        energy_loss = (bce_criterion(masked_y_pred, masked_y_true)\
                      +bce_criterion(masked_y_trivial_pred,masked_y_trivial_true))/2 + args.lmd_1 * matrix_loss
  
    else:
        y_pred,A_causual = prediction_function( 
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, mask=mask,
            use_trivial=use_trivial,args=args
        ) # B, T, 1
        y_true = y_true.unsqueeze(2)  # B, T, 1
        y_valid = y_true ** 2 > 0
        y_true = ((1 + y_true) / 2)  # B, T, 1

        masked_y_true = torch.masked_select(y_true, y_valid)
        masked_y_pred = torch.masked_select(y_pred, y_valid)
        energy_loss = bce_criterion(masked_y_pred, masked_y_true)
    return energy_loss,A_causual





def extract_log_prob_for_1st_order(y_pred_1st_order, y_true, y_valid, task_edge, args):
    '''
    :param y_pred_1st_order: (B, T, 2)
    :param y_pred_2nd_order: (B, M, 4)
    :param y_true: B, T
    :param y_valid: B, T
    :return:
    '''
    y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
    y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
    y_pred_1st_order = torch.gather(y_pred_1st_order, 2, y_true_1st_order)  # (B, T, 2) => (B, T, 1)
    y_pred_1st_order = torch.log(y_pred_1st_order + EPS)
    energy_1st_order = torch.sum(y_pred_1st_order*y_valid_1st_order, dim=1)  # B, 1

    energy = energy_1st_order.squeeze(1)

    return energy


    
def extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args):
    '''
    :param y_pred_1st_order: (B, T, 2) softmax --> (0,1)
    :param y_pred_2nd_order: (B, M, 4)
    :param y_true: B, T
    :param y_valid: B, T
    :return:
    '''

    y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
    y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
    y_pred_1st_order = torch.gather(y_pred_1st_order, 2, y_true_1st_order)  # (B, T, 2) => (B, T, 1)
    y_pred_1st_order = torch.log(y_pred_1st_order + EPS)

    energy_1st_order = torch.sum(y_pred_1st_order * y_valid_1st_order, dim=1)  # B, 1

    y_true_2nd_order = mapping_label_02(y_true, task_edge)  # B, M, 1

    y_valid_2nd_order = mapping_valid_label(y_valid, task_edge)  # B, M, 1
    y_pred_2nd_order = torch.gather(y_pred_2nd_order, 2, y_true_2nd_order)  # (B, M, 4) => (B, M, 1)
    y_pred_2nd_order = torch.log(y_pred_2nd_order + EPS)
    energy_2nd_order = torch.sum(y_pred_2nd_order * y_valid_2nd_order, dim=1)  # B, 1

    energy = energy_1st_order + args.structured_lambda * energy_2nd_order
    energy = energy.squeeze(1)
    return energy


def passing(x, x_2nd_agg, edge):
    T = x.size()[1]
    node_in, node_out = edge[0], edge[1]  # M, M

    update = (scatter_add(x_2nd_agg, node_out, dim=1, dim_size=T) +
              scatter_add(x_2nd_agg, node_in, dim=1, dim_size=T)) / 2  # B, T, d
    x = x + update  # B, T, d

    return x

def mix_passing(fi, fij, edge):
    aggregate_indice = torch.LongTensor([0, 0, 1, 1]).to(args.device)
    x_1st_neg = fi[:, :, 0]  # B, T, 1
    x_1st_pos = fi[:, :, 1]  # B, T, 1

    x_2nd_agg = scatter_add(fij, aggregate_indice, dim=2)  # B, T, 2
    x_2nd_neg = x_2nd_agg[:, :, 0]  # B, M, d
    x_2nd_pos = x_2nd_agg[:, :, 1]  # B, M, d

    x_neg = passing(x_1st_neg, x_2nd_neg, edge)  # B, T, d
    x_pos = passing(x_1st_pos, x_2nd_pos, edge)
    fi_mix = torch.stack([x_neg, x_pos], dim=2)  # B, T, 2
    fi_mix = softmax_opt(fi_mix)
    return fi_mix

def energy_function_output(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge, args, mask, use_trivial,A_causual, DAG, adj,
        prior_prediction=None, prior_prediction_logits=None, id=None, **kwargs):

    y_pred_1st_order, y_pred_2nd_order = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model, mask=mask, use_trivial=use_trivial,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, adj=adj,
        A_causual=A_causual, args=args
    )

    B = y_pred_1st_order.size()[0]
    M = y_pred_2nd_order.size()[1]
    y_valid = y_true ** 2 > 0  # B, T
    y_true = ((1+y_true) / 2).long()  # B, T


    if args.NCE_mode == 'uniform':
        # TODO: filling 0 for missing labels
        y_valid = (y_valid >= -1).long()

        x_noise_1st_order = torch.ones_like(y_pred_1st_order) * 0.5  # B, M, 2
        x_noise_2nd_order = torch.ones_like(y_pred_2nd_order) * 0.25  # B, M, 4

        log_p_x = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args)  # B
        log_q_x = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_true, y_valid, task_edge, args)  # B, -T log(2)
        # # TODO:
        # import math
        # T = y_pred_1st_order.size()[1]
        # print('log_q_x\t', log_q_x, '\t', T * math.log(2))
        loss_data = log_p_x - torch.logsumexp(torch.stack([log_p_x, log_q_x], dim=1), dim=1, keepdim=True)

        y_noise = softmax_opt(torch.rand_like(y_pred_1st_order))  # B, T, 2
        y_noise = (y_noise[..., 1] >= 0.5).long()  # B, T
        log_p_noise = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_noise, y_valid, task_edge, args)  # B
        log_q_noise = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_noise, y_valid, task_edge, args)  # B, -T log(2)
        loss_noise = log_q_noise - torch.logsumexp(torch.stack([log_p_noise, log_q_noise], dim=1), dim=1, keepdim=True)

        loss = - (loss_data.mean() + loss_noise.mean())

    elif args.NCE_mode == 'gs':
        x_noise_1st_order = output_GS_inference(
            y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order, return_full_prob=True,
            first_order_prediction_model=first_order_prediction_model,use_trivial=use_trivial,
            second_order_prediction_model=second_order_prediction_model,mask=mask,adj=adj,
            GNN_energy_model=GNN_energy_model, prediction_function=prediction_function,A_causual=A_causual,
            prior_prediction=prior_prediction, prior_prediction_logits=prior_prediction_logits, id=id,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, args=args
        )  # B, T, 2

        y_prob = torch.rand_like(x_noise_1st_order)  # B, T, 2
        y_noise = (x_noise_1st_order >= y_prob).long()[..., 1]  # B, T
        if args.filling_missing_data_mode is not 'no_filling':
            y_true = torch.where(y_valid, y_true, y_noise)
            y_valid.fill_(1)

        log_p_x = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args)  # B
        log_q_x = extract_log_prob_for_1st_order(x_noise_1st_order, y_true, y_valid, task_edge, args)  # B
        loss_data = log_p_x - torch.logsumexp(torch.stack([log_p_x, log_q_x], dim=1), dim=1, keepdim=True)


        log_p_noise = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_noise, y_valid, task_edge, args)  # B
        log_q_noise = extract_log_prob_for_1st_order(x_noise_1st_order, y_noise, y_valid, task_edge, args)  # B
        loss_noise = log_q_noise - torch.logsumexp(torch.stack([log_p_noise, log_q_noise], dim=1), dim=1, keepdim=True)

        loss = - (loss_data.mean() + loss_noise.mean())

    elif args.NCE_mode == 'do_operation':

        fi_mix = mix_passing(y_pred_1st_order, y_pred_2nd_order, task_edge)  # done softmax
        dag_loss = torch.nn.MSELoss(reduction='mean')
        fi_g = DAG(fi_mix)
        loss_dag = dag_loss(fi_g, fi_mix)

        fi_mix_noise = fi_mix.clone().detach()
        neg = torch.full((B,50,2),0.3).to(args.device)
        pos = torch.ones(B,50,2).to(args.device)

        fi_mix_noise[:, 1:51, :] = torch.where(fi_mix_noise[:, 1:51, :] > 0.5, neg, pos)

        fi_mix_noise_new = DAG(fi_mix_noise)

        x_noise_1st_order = output_GS_inference(
            y_pred_1st_order=fi_mix_noise_new, y_pred_2nd_order=y_pred_2nd_order, return_full_prob=True,
            first_order_prediction_model=first_order_prediction_model,use_trivial=use_trivial,
            second_order_prediction_model=second_order_prediction_model,mask=mask,adj=adj,
            GNN_energy_model=GNN_energy_model, prediction_function=prediction_function,A_causual=A_causual,
            prior_prediction=prior_prediction, prior_prediction_logits=prior_prediction_logits, id=id,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, args=args
        )  # B, T, 2

        y_prob = torch.rand_like(x_noise_1st_order)  # B, T, 2
        y_noise = (x_noise_1st_order >= y_prob).long()[..., 1]  # B, T
        if args.filling_missing_data_mode is not 'no_filling':
            y_true = torch.where(y_valid, y_true, y_noise)
            y_valid.fill_(1)

        dag_param = DAG.adj

        h_a = _h_A(dag_param, dag_param.size()[0], args)
        loss_h_a = 100 * h_a + 10 * h_a * h_a

        y_prob = torch.rand_like(fi_mix_noise)  # B, T, 2
        y_noise = (fi_mix_noise >= y_prob).long()[..., 1]  # B, T

        if args.filling_missing_data_mode is not 'no_filling':
            y_true = torch.where(y_valid, y_true, y_noise)
            y_valid.fill_(1)

        log_p_x = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args)  # B
        log_q_x = extract_log_prob_for_1st_order(x_noise_1st_order, y_true, y_valid, task_edge,args)  # B
        loss_data = log_p_x - torch.logsumexp(torch.stack([log_p_x, log_q_x], dim=1), dim=1, keepdim=True)
        log_p_noise = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_noise, y_valid, task_edge,args)  # B
        log_q_noise = extract_log_prob_for_1st_order(x_noise_1st_order, y_noise, y_valid, task_edge,args)  # B
        loss_noise = log_q_noise - torch.logsumexp(torch.stack([log_p_noise, log_q_noise], dim=1), dim=1, keepdim=True)
        loss = - (loss_data.mean() + loss_noise.mean())
        loss = loss + args.lmd_2 * loss_dag + args.lmd_3 * loss_h_a

    else:
        raise ValueError('NCE mode {} not included.'.format(args.NCE_mode))

    return loss





# ----------------------------------------------------------------------------------------------------------------------
#
# Inference Function
#
# ----------------------------------------------------------------------------------------------------------------------

def feature_inference(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, task_edge, mask, args, **kwargs):
    with torch.no_grad():
        if args.energy_function == 'energy_function_feature':
            y_pred_1st, A_causual = prediction_function(
                first_order_prediction_model=first_order_prediction_model,
                second_order_prediction_model=second_order_prediction_model,
                GNN_energy_model=GNN_energy_model,mask=mask,use_trivial=False,
                graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
                args=args
            )
            y_pred = y_pred_1st.squeeze(2)  # B, T, 1  ->  B, T
        else:
            y_pred_1st, _ = prediction_function(
                first_order_prediction_model=first_order_prediction_model,
                second_order_prediction_model=second_order_prediction_model,
                GNN_energy_model=GNN_energy_model,use_trivial=False,
                graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
                args=args
            )
            y_pred = y_pred_1st[..., 1]  # (B, T, 2) ===> (B, T)

    return y_pred


def GNN_EBM_mean_field_variational_inference(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, task_edge, args,
        y_pred_1st_order=None, y_pred_2nd_order=None, return_full_prob=False, **kwargs):
    '''
    :return: logits or confidence: B, T
    '''
    if y_pred_1st_order is None:
        y_pred_1st_order, y_pred_2nd_order = prediction_function(
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
            args=args)
    B = y_pred_1st_order.size()[0]
    T = y_pred_1st_order.size()[1]
    M = y_pred_2nd_order.size()[1]

    T = y_pred_1st_order.size()[1]

    y_prior = None
    if args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm'] and args.ebm_as_tilting:
        prior_prediction_logits = kwargs['prior_prediction_logits']
        y_prior = prior_prediction_logits.detach().clone()
        y_prior = y_prior[kwargs['id']]
    # p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    q = torch.abs(torch.randn_like(y_pred_1st_order))  # B, T, 2
    q = softmax_opt(q)  # B, T, 2

    aggregate_indice = torch.LongTensor([0, 0, 1, 1]).to(args.device)

    for _ in range(args.MFVI_iteration):
        q_i, q_j = q[:, task_edge[0]], q[:, task_edge[1]]  # (B,T,2) => (B,M,2)
        q_second_order = torch.einsum('bmxy,bmyz->bmxz', q_i.unsqueeze(3), q_j.unsqueeze(2))  # B, M, 2, 2
        q_second_order = q_second_order.view(B, M, 4)  # B, M, 4

        aggregated_y_pred_2nd_order = scatter_add(q_second_order*y_pred_2nd_order, task_edge[0], dim=1, dim_size=T)  # B, T, 4
        y_pred_2nd_order_ = scatter_add(aggregated_y_pred_2nd_order, aggregate_indice, dim=2)  # B, T, 2

        if y_prior is not None and args.ebm_as_tilting:
            prior = F.sigmoid(torch.stack([-y_prior, y_prior], dim=2))
        else:
            prior = 1
        q = y_pred_1st_order + args.structured_lambda * y_pred_2nd_order_
        q = softmax_opt(q) * prior

    if return_full_prob:
        return q
    return q[..., 1]


def output_GS_inference(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, task_edge, args, mask, A_causual, adj,
        y_pred_1st_order=None, y_pred_2nd_order=None, return_full_prob=False, **kwargs):
    '''
    :return: logits or confidence: B, T
    '''
    if y_pred_1st_order is None:
        y_pred_1st_order, y_pred_2nd_order = prediction_function(
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,mask=mask,use_trivial=False,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
            A_causual=A_causual,adj=adj,args=args
        )


    T = y_pred_1st_order.size()[1]
    if args.filling_missing_data_mode == 'no_filling':
        y_sample = ((softmax_opt(y_pred_1st_order)[..., 1]) > 0.5).long()  # B, T
    elif args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm']:
        prior_prediction = kwargs['prior_prediction']
        y_sample = prior_prediction.detach().clone()
        y_sample = y_sample[kwargs['id']]
    else:
        raise ValueError('{} not implemented.'.format(args.filling_missing_data_mode))

    y_prior = None
    if args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm'] and args.ebm_as_tilting:
        prior_prediction_logits = kwargs['prior_prediction_logits']
        y_prior = prior_prediction_logits.detach().clone()
        y_prior = y_prior[kwargs['id']]
    p = torch.rand_like(y_pred_1st_order) 
    p_accum = 0

    for layer_idx in range(args.GS_iteration):
        y_sample, p = Gibbs_sampling(
            T=T, y_sample=y_sample, p=p, y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order,
            task_edge=task_edge, args=args, y_prior=y_prior)

        p_accum += p.clone()

    if args.GS_inference == 'last':
        p_accum = p
    elif args.GS_inference == 'average':
        p_accum /= args.GS_iteration #B,T,2

    if return_full_prob:
        return p_accum
    return p_accum[..., 1] # B,T,1


def GNN_EBM_1st_order_inference_Binary_Task(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge, args, **kwargs):
    y_pred, _ = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    y_pred = y_pred[..., 1]  # (B, T, 2) ===> (B, T)
    return y_pred
