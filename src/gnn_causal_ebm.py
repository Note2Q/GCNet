import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from configure import *

bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
ce_criterion = nn.CrossEntropyLoss()
softmax_opt = nn.Softmax(-1)
sigmoid_opt = nn.Sigmoid()


def mapping_task_repr(task_repr, task_edge, B):
    '''
    Mapping task_repr to node1_task_repr and node2_task_repr, w.r.t. task_edge
    :param task_repr: T, d_task
    :param task_edge:  2, M
    :return: (B, M, d_task), (B, M, d_task)
    '''
    task_repr_2nd_order_node1 = torch.index_select(task_repr, 0, task_edge[0])
    task_repr_2nd_order_node2 = torch.index_select(task_repr, 0, task_edge[1])
    task_repr_2nd_order_node1 = task_repr_2nd_order_node1.unsqueeze(0).expand(B, -1, -1)  # B, M, d_task
    task_repr_2nd_order_node2 = task_repr_2nd_order_node2.unsqueeze(0).expand(B, -1, -1)  # B, M, d_task
    return task_repr_2nd_order_node1, task_repr_2nd_order_node2


def mapping_label(y_true, task_edge):
    '''
    [-1, -1] => 0
    [-1,  1] => 1
    [ 1, -1] => 2
    [ 1,  1] => 3
    '''
    y_true_node1 = torch.index_select(y_true, 1, task_edge[0])
    y_true_node2 = torch.index_select(y_true, 1, task_edge[1])

    y_true_2nd_order = ((2 * y_true_node1 + y_true_node2 + 3) / 2).long().unsqueeze(2)
    return y_true_2nd_order


def mapping_label_02(y_true, task_edge):
    '''
    [0, 0] => 0
    [0, 1] => 1
    [1, 0] => 2
    [1, 1] => 3
    '''
    y_true_node1 = torch.index_select(y_true, 1, task_edge[0])
    y_true_node2 = torch.index_select(y_true, 1, task_edge[1])

    y_true_2nd_order = (2 * y_true_node1 + y_true_node2).long().unsqueeze(2)
    return y_true_2nd_order


def mapping_valid_label(y_valid, task_edge):
    y_valid_node1 = torch.index_select(y_valid, 1, task_edge[0])
    y_valid_node2 = torch.index_select(y_valid, 1, task_edge[1])

    y_valid_2nd_order = torch.logical_and(y_valid_node1, y_valid_node2).unsqueeze(2)
    return y_valid_2nd_order


def extract_amortized_task_label_weights(dataloader, task_edge, device, args):
    '''
    :return:
    first_order_label_weights (T, 2)
    second_order_label_weights (M, 4)
    '''
    T = args.num_tasks
    M = len(task_edge[0])
    print('T={} tasks, M={} task edges.'.format(T, M))
    first_order_label_weights = torch.zeros((T, 2), device=device)
    second_order_label_weights = torch.zeros((M, 4), device=device)
    first_order_valid_counts = torch.zeros(T, device=device)
    second_order_valid_counts = torch.zeros(M, device=device)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            B = len(batch.id)
            y_true = batch.y.view(B, args.num_tasks).float()  # B, T
            y_true_first_order = ((1 + y_true) / 2).long()  # B, T
            y_valid_first_order = y_true ** 2 > 0  # B, T

            y_true_second_order = mapping_label(y_true, task_edge).squeeze()  # B, M
            y_valid_second_order = mapping_valid_label(y_valid_first_order, task_edge).squeeze()  # B, M

            for label in [0, 1]:
                batch_statistics = torch.logical_and(y_true_first_order == label, y_valid_first_order)
                first_order_label_weights[:, label] += batch_statistics.sum(0)

            for label in [0, 1, 2, 3]:
                batch_statistics = torch.logical_and(y_true_second_order == label, y_valid_second_order)
                second_order_label_weights[:, label] += batch_statistics.sum(0)

            first_order_valid_counts += y_valid_first_order.sum(0)
            second_order_valid_counts += y_valid_second_order.sum(0)

    for i in range(T):
        assert first_order_label_weights[i].sum(0) == first_order_valid_counts[i]

    for i in range(M):
        assert second_order_label_weights[i].sum(0) == second_order_valid_counts[i]

    first_order_label_weights /= first_order_valid_counts.unsqueeze(1)
    second_order_label_weights /= second_order_valid_counts.unsqueeze(1)

    return first_order_label_weights, second_order_label_weights


def get_EBM_prediction(
        first_order_prediction_model, second_order_prediction_model,
        graph_repr, task_repr, task_edge, args):
    B = len(graph_repr)
    M = len(task_edge[0])

    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order = task_repr.unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st_order = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order)  # B, T, 2

    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol
    task_repr_2nd_order_node1, task_repr_2nd_order_node2 = mapping_task_repr(task_repr, task_edge,
                                                                             B)  # (B, M, d_task), (B, M, d_task)
    y_pred_2nd_order = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1,
                                                     task_repr_2nd_order_node2)  # B, M, 4

    return y_pred_1st_order, y_pred_2nd_order




def Gibbs_sampling(T, y_sample, p, y_pred_1st_order, y_pred_2nd_order, task_edge, args, y_prior=None, **kwargs):
    def gather(y_sample_edge, filtered_y_pred_2nd_order):
        '''
        :param y_sample_edge: (B, k, 1)
        :param filtered_y_pred_2nd_order: (B, k, 4)
        :return:
        '''
        y_pred_2nd_order_ = torch.gather(filtered_y_pred_2nd_order, 2, y_sample_edge)  # (B, k, 4) (B, k, 1) => (B, k, 1)
        y_pred_2nd_order_ = y_pred_2nd_order_.sum(dim=1)  # B, 1
        return y_pred_2nd_order_

    with torch.no_grad():
        task_idx_list = torch.randperm(T)  
        for i in task_idx_list:
            filtered_task_edge_index = (task_edge[0] == i).nonzero(as_tuple=False) # k, 1
            filtered_y_pred_2nd_order = y_pred_2nd_order[:,filtered_task_edge_index.squeeze(1)]  # (B, T, 4) => (B, k, 4)

            y_sample_j_index = task_edge[1][filtered_task_edge_index].squeeze(1)  # k
            y_sample_j_label = (y_sample[:, y_sample_j_index]).unsqueeze(2)  # B, k, 1

            y_sample_edge_label_0 = y_sample_j_label  # B, k, 1 [0,1]
            y_sample_edge_label_1 = 2 + y_sample_j_label  # B, k, 1 [2,3]

            y_sample_pred_edge_0j = gather(y_sample_edge_label_0, filtered_y_pred_2nd_order)  # B, 1
            y_sample_pred_label_1j = gather(y_sample_edge_label_1, filtered_y_pred_2nd_order)  # B, 1
            y_pred_2nd_order_ = torch.cat([y_sample_pred_edge_0j, y_sample_pred_label_1j], dim=1)  # B, 2

            temp = y_pred_1st_order[:, i] + args.structured_lambda * y_pred_2nd_order_  # B, 2
            temp = softmax_opt(temp)  # B, 2
            if y_prior is not None and args.ebm_as_tilting:
                prior = torch.sigmoid(torch.stack([-y_prior[:, i], y_prior[:, i]], dim=1))
                temp *= prior

            p[:, i] = temp
            y_sample[:, i] = (temp[..., 1] > torch.rand_like(temp[..., 1])).long()  # B

    return y_sample, p


def GS_inference(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''

    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    T = y_pred_1st_order.size()[1]
    if args.filling_missing_data_mode == 'no_filling':
        y_sample = ((softmax_opt(y_pred_1st_order)[..., 1]) > 0.5).long()  # B, T
    else:
        raise ValueError('{} not implemented.'.format(args.filling_missing_data_mode))
    p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    p_accum = 0

    for layer_idx in range(args.GS_iteration):
        y_sample, p = Gibbs_sampling(
            T=T, y_sample=y_sample, p=p, y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order,
            task_edge=task_edge, args=args)

        p_accum += p.clone()

    if args.GS_inference == 'last':
        p_accum = p
    elif args.GS_inference == 'average':
        p_accum /= args.GS_iteration
    return p_accum[..., 1]



if __name__ == '__main__':
    edge = torch.LongTensor(
        [
            [0, 1, 1, 2, 0, 3],
            [1, 0, 2, 1, 3, 0]
        ]
    )
    x = torch.LongTensor([[0, 1, 2, 3, 4, 5]])
    x = torch.LongTensor([[0, 0, 2, 2, 4, 4]])
    print(x, '\n')

    print(edge[0])
    print(edge[1])

    ans = scatter_add(x, edge[0], dim=1)
    print('scatter along 0\t', ans, '\n')

    ans = scatter_add(x, edge[1], dim=1)
    print('scatter along 1\t', ans, '\n')
