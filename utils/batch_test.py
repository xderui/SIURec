import torch
import numpy as np
from torch.nn import functional as F

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def Recall_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    return recall


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def ILD(item_embeddings):
    return torch.pdist(item_embeddings,p=2).mean()

def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    recall, ndcg = [], []
    for k in topks:
        recall.append(Recall_ATk(groundTrue, r, k))
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def eval_PyTorch(model, data_generator, Ks):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    test_users = list(data_generator.test_dict.keys())

    u_batch_size = data_generator.batch_size

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    batch_rating_list = []
    ground_truth_list = []
    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rate_batch = model.predict(user_batch)

        count += rate_batch.shape[0]

        exclude_index = []
        exclude_items = []
        ground_truth = []
        for i in range(len(user_batch)):
            train_dict = list(data_generator.train_dict[user_batch[i]])
            exclude_index.extend([i] * len(train_dict))
            exclude_items.extend(train_dict)
            ground_truth.append(list(data_generator.test_dict[user_batch[i]]))
        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        batch_rating_list.append(rate_batch_k.cpu())
        ground_truth_list.append(ground_truth)

    # # gini
    # item_num = ia_embeddings.shape[0]
    # cnt = [0 for i in range(item_num)]

    X = zip(batch_rating_list, ground_truth_list)
    batch_results = []
    diversity = []
    for x in X:
        batch_results.append(test_one_batch(x, Ks))
        # predict_user_items = x[0]
        # for predict_items in predict_user_items:
        #     predict_embeddings = ia_embeddings[predict_items]
    #         diversity.append(ILD(F.normalize(predict_embeddings, dim=-1)))
    #         for item_ in predict_items:
    #             cnt[item_] += 1

    # diversity = torch.mean(torch.tensor(diversity)).item()

    for batch_result in batch_results:
        result['recall'] += batch_result['recall'] / n_test_users
        result['ndcg'] += batch_result['ndcg'] / n_test_users

    # result['diversity'] = diversity

    # max_id = np.argmax(cnt)
    # print(max_id, cnt[max_id])

    # cnt.sort()
    # giny = 0
    # height, area = 0, 0
    # for c in cnt:
    #     height += c
    #     area += height-c/2
    # fair_area = height*item_num/2
    # giny = (fair_area-area)/fair_area
    
    

    result['diversity'] = -1
    result['gini'] = -1

    assert count == n_test_users

    return result