import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse



class SIURec(nn.Module):
    def __init__(self, args, adj_mat):
        super(SIURec, self).__init__()
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.rows = list(adj_mat.tocoo().row)
        self.cols = list(adj_mat.tocoo().col)
        self.values = list(adj_mat.tocoo().data)

        self.A_shape = adj_mat.tocoo().shape
        self.A_indices = torch.tensor([self.rows, self.cols], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.num_users + self.num_items)), list(range(self.num_users + self.num_items))], dtype=torch.long).cuda()
        
        self.rows = torch.LongTensor(self.rows).cuda()
        self.cols = torch.LongTensor(self.cols).cuda()
        self.G_indices, self.G_values = self.laplacian_adj()

        self.emb_size = args.embed_size
        self.n_layers = args.n_layers

        self.emb_reg = args.emb_reg


        self.embedding_user = nn.Embedding(self.num_users, self.emb_size)
        self.embedding_item = nn.Embedding(self.num_items, self.emb_size)

        # init
        nn.init.xavier_normal_(self.embedding_user.weight)
        nn.init.xavier_normal_(self.embedding_item.weight)


    def laplacian_adj(self):
        
        A_values = torch.ones((len(self.rows), 1)).view(-1).cuda()

        A_indices_sparse = torch_sparse.SparseTensor(row=self.rows, col=self.cols, value=A_values, sparse_sizes=self.A_shape).cuda()

        D_values = A_indices_sparse.sum(dim=-1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_shape[0], self.A_shape[1], self.A_shape[1])

        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_shape[0], self.A_shape[1], self.A_shape[1])

        return G_indices, G_values
    
    def fine_laplacian_adj(self, gnn_embs):
        gnn_embs_row = torch.index_select(gnn_embs, 0, self.rows)
        gnn_embs_col = torch.index_select(gnn_embs, 0, self.cols)

        gnn_embs_row = F.normalize(gnn_embs_row)
        gnn_embs_col = F.normalize(gnn_embs_col)

        scores = (torch.sum(gnn_embs_row * gnn_embs_col, dim=1).view(-1) + 1) / 2

        A_indices_sparse = torch_sparse.SparseTensor(row=self.rows, col=self.cols, value=scores, sparse_sizes=self.A_shape).cuda()

        D_values = A_indices_sparse.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.rows, self.cols], dim=0)
        G_values = D_values[self.rows] * scores

        return G_indices, G_values

    def forward(self):
        all_embs = [torch.concat([self.embedding_user.weight, self.embedding_item.weight], dim=0)]

        user_fine_interests = []
        item_fine_interests = []

        for i in range(self.n_layers):
            gnn_embs = torch_sparse.spmm(self.G_indices, self.G_values, self.A_shape[0], self.A_shape[1], all_embs[i])

            fine_G_indices, fine_G_values = self.fine_laplacian_adj(gnn_embs)
            fine_interests_embs = torch_sparse.spmm(fine_G_indices, fine_G_values, self.A_shape[0], self.A_shape[1], all_embs[i])

            # fine_interests.append(fine_interests_embs)

            user_fine_interests_embs, item_fine_interests_embs = torch.split(fine_interests_embs, [self.num_users, self.num_items], 0)

            all_embs.append(gnn_embs + fine_interests_embs)
            user_fine_interests.append(user_fine_interests_embs)
            item_fine_interests.append(item_fine_interests_embs)

        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.sum(all_embs, dim=1, keepdim=False)

        self.u_embs, self.i_embs = torch.split(all_embs, [self.num_users, self.num_items], dim=0)

        return self.u_embs, self.i_embs, user_fine_interests, item_fine_interests


    def predict(self, users):
        u_embs = self.u_embs[torch.LongTensor(users).cuda()]
        i_embs = self.i_embs
        batch_ratings = torch.matmul(u_embs, i_embs.T)
        return batch_ratings
    
    def get_ego_embedding(self):
        return self.embedding_user, self.embedding_item