import torch
import numpy as np 
from utils.load_data import Data
from utils.parser import parse_args
from model.SIURec import SIURec
from utils.loss import *
import scipy.optimize as opt
from utils.batch_test import *
from tqdm import tqdm
import time

def pareto_efficient_weights(prev_w, c, G):
    GGT = np.matmul(G, np.transpose(G))  # [K, K]
    e = np.ones(np.shape(prev_w))  # [K, 1]
    m_up = np.hstack((GGT, e))  # [K, K+1]
    m_down = np.hstack((np.transpose(e), np.zeros((1, 1))))  # [1, K+1]
    M = np.vstack((m_up, m_down))  # [K+1, K+1]

    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # [K+1, 1]

    MTM = np.matmul(np.transpose(M), M)
    w_hat = np.matmul(np.matmul(np.linalg.inv(MTM), M), z)  # [K+1, 1]
    w_hat = w_hat[:-1]  # [K, 1]
    w_hat = np.reshape(w_hat, (w_hat.shape[0],))  # [K,]

    return active_set_method(w_hat, prev_w.squeeze(-1), c)


if __name__ == "__main__":

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # Prepare data

    data = Data(args)

    adj_mat = data.adj_mat()
    
    # model
    args.num_users = data.num_users
    args.num_items = data.num_items
    model = SIURec(args, adj_mat).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    w_uniform = [1] * args.n_layers
    w_bpr_uniform = [1,1]
    mean_w_uniform = []
    mean_w_bpr_uniform = []
    bpr_uniform_grad_mean = [[],[]]

    for epoch in range(args.epoch):
        batches = data.uniform_sample()

        model.train()

        total_loss = 0
        total_bpr_loss = 0
        total_reg_loss = 0
        total_uniform_loss = 0

        for idx in tqdm(range(args.n_batch)):
            optimizer.zero_grad()

            users, pos_items, neg_items = data.batch_data(idx)
            users = torch.LongTensor(users).cuda()
            pos_items = torch.LongTensor(pos_items).cuda()
            neg_items = torch.LongTensor(neg_items).cuda()

            u_embs, i_embs, u_fine_interests, i_fine_interests = model()
            u_embs_ = u_embs[users]
            pos_embs_ = i_embs[pos_items]
            neg_embs_ = i_embs[neg_items]


            # bpr loss

            bpr_loss = BPR_LOSS(u_embs_, pos_embs_, neg_embs_)

            # reg loss
            ego_u_embs, ego_i_embs = model.get_ego_embedding()
            ego_u_embs = ego_u_embs(users)
            ego_pos_embs = ego_i_embs(pos_items)
            ego_neg_embs = ego_i_embs(neg_items)
            reg_loss = REG_LOSS_POW(args.emb_reg, ego_u_embs, ego_pos_embs, ego_neg_embs)

            # sub-interests uniformity
            uniform_losses = []
            uniform_loss = 0
            unique_users = torch.unique(users)
            unique_items = torch.unique(pos_items)
            for idx, (u_interests, i_interests) in enumerate(zip(u_fine_interests, i_fine_interests)):
                uniform_losses.append(args.uniform_reg * (UNIFORM_LOSS(u_interests[unique_users]) + UNIFORM_LOSS(i_interests[unique_items])))
                uniform_loss = uniform_loss + w_uniform[idx] * uniform_losses[idx]


            batch_loss = bpr_loss * w_bpr_uniform[0] + reg_loss + uniform_loss * w_bpr_uniform[1]

            total_loss += batch_loss
            total_bpr_loss += bpr_loss
            total_reg_loss += reg_loss
            total_uniform_loss += uniform_loss


            parameters = list(model.parameters())
            user_item_idx = [users, pos_items]
            if epoch == 0:
                layers_grads = []
                for i, layer_loss in enumerate(uniform_losses):
                    layer_params_grad = list(torch.autograd.grad(layer_loss, parameters, retain_graph=True))
                    for idx, grad_ in enumerate(layer_params_grad):
                        layer_params_grad[idx] = grad_[user_item_idx[idx]]
                    layer_params_grad = torch.concat(layer_params_grad, dim=0)
                    layers_grads.append(layer_params_grad)

                layers_grads = [v.view(-1) for v in layers_grads]
                G = torch.stack(layers_grads, dim=0).cpu().detach().numpy()
                w_uniform_temp = pareto_efficient_weights(prev_w=np.asarray([1]*args.n_layers).reshape(args.n_layers,1),
                    c=np.array([0.0]*args.n_layers).reshape(args.n_layers,1),  
                    G=G)
                
                mean_w_uniform.append(w_uniform_temp)

            
                # align grad
                layers_grads = []
                layer_params_grad = list(torch.autograd.grad(bpr_loss / 0.6943, parameters, retain_graph=True ))  # use the value of the embeddings when they were first initialised as the upper bound of BPR loss
                for idx, grad_ in enumerate(layer_params_grad):
                    layer_params_grad[idx] = grad_[user_item_idx[idx]]
                layer_params_grad = torch.concat(layer_params_grad, dim=0)
                layers_grads.append(layer_params_grad)

                #uniform grad
                layer_params_grad = list(torch.autograd.grad((uniform_loss+7.48*args.ssl_reg*args.n_layers) / (7.48*args.ssl_reg*args.n_layers), parameters, retain_graph=True ))
                for idx, grad_ in enumerate(layer_params_grad):
                    layer_params_grad[idx] = grad_[user_item_idx[idx]]
                layer_params_grad = torch.concat(layer_params_grad, dim=0)
                layers_grads.append(layer_params_grad)

                layers_grads = [v.view(-1) for v in layers_grads]
                G = torch.stack(layers_grads, dim=0).cpu().detach().numpy()

                w_bpr_uniform_temp = pareto_efficient_weights(prev_w=np.asarray([1]*2).reshape(2,1),
                    c=np.array([0.0]*2).reshape(2,1),
                    G=G)

                bpr_uniform_grad_mean[0].append(torch.mean(torch.abs(layers_grads[0])).item())
                bpr_uniform_grad_mean[1].append(torch.mean(torch.abs(layers_grads[1])).item())
                
                mean_w_bpr_uniform.append(w_bpr_uniform_temp)


            # t1 = time.time()
            batch_loss.backward()
            optimizer.step()
            # print(time.time()-t1)

        if epoch == 0:
            w_bpr_uniform_temp = np.mean(mean_w_bpr_uniform, axis=0)
            w_uniform_temp = np.mean(mean_w_uniform, axis=0)

            max_id = np.argmax(w_uniform_temp)
            w_uniform[max_id] = 1
            for idx in range(args.n_layers):
                if idx != max_id:
                    w_uniform[idx] =  w_uniform[max_id] * w_uniform_temp[idx] / w_uniform_temp[max_id]

            bpr_uniform_grad_mean = [np.mean(vlist) for vlist in bpr_uniform_grad_mean]
            w_bpr_uniform[1] = w_bpr_uniform[0] * bpr_uniform_grad_mean[0] / bpr_uniform_grad_mean[1] / args.uniform_reg

            print(w_bpr_uniform, w_uniform)

        torch.cuda.empty_cache()
        
        total_loss /= args.n_batch
        total_bpr_loss /= args.n_batch
        total_reg_loss /= args.n_batch
        total_uniform_loss /= args.n_batch

        # evaluate
        with torch.no_grad():
            model.eval()
            model()
            test_ret = eval_PyTorch(model, data, eval(args.topk))

        output_train = 'Epoch %d: train_loss = [%.5f=%.5f + %.5f + %.5f]' % (epoch, total_loss, total_bpr_loss, total_reg_loss, total_uniform_loss)
        output_val = ''
        for k,v in test_ret.items():
            output_val += f', val-{k}=[{str([round(v_, 4) for v_ in v])}]'
        
        print(output_train + output_val)
