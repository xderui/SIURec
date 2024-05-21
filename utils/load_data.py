import os
import pickle
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

class Data():
    def __init__(self, args):
        self.data_root = args.data_root
        self.data_path = os.path.join(self.data_root, args.dataset)
        self.batch_size = args.batch_size
        self.n_batch = args.n_batch

        train_file = os.path.join(self.data_path, 'train.pkl')
        val_file = os.path.join(self.data_path, 'val.pkl')
        test_file = os.path.join(self.data_path, 'test.pkl')

        self.train_mat = pickle.load(open(train_file, 'rb'))
        self.val_mat = pickle.load(open(val_file, 'rb'))
        self.test_mat = pickle.load(open(test_file, 'rb'))
        
        self.num_users, self.num_items = self.train_mat.shape

        # user-item dict of train_mat
        self.train_dict = self.construct_dict(self.train_mat)
        self.val_dict = self.construct_dict(self.val_mat)
        self.test_dict = self.construct_dict(self.test_mat)

                
    def construct_dict(self, mat):
        users, items = mat.row, mat.col
        dict_ = defaultdict(list)
        for idx in range(len(users)):
            user = users[idx]
            item = items[idx]
            dict_[user].append(item)

        return dict_


    def adj_mat(self):
        A_mat = self.train_mat.todok()
        rows = A_mat.tocoo().row
        cols = A_mat.tocoo().col

        rows_ = np.concatenate([rows, cols+self.num_users], axis=0)
        cols_ = np.concatenate([cols+self.num_users, rows], axis=0)

        adj_mat = sp.coo_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=[self.num_users + self.num_items, self.num_users + self.num_items]).tocsr()

        return adj_mat
    
    def uniform_sample(self):
        users = np.random.randint(0, self.num_users, self.n_batch*self.batch_size)

        train_data = []
        for user in users:
            pos_item = np.random.choice(self.train_dict[user])
            neg_item = np.random.randint(0, self.num_items)
            while neg_item in self.train_dict[user]:
                neg_item = np.random.randint(0, self.num_items)

            train_data.append([user, pos_item, neg_item])

        self.train_data = np.array(train_data)

    
    def batch_data(self, batch_idx):
        batch_data_ = self.train_data[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
        return batch_data_[:, 0], batch_data_[:, 1], batch_data_[:, 2]
    
    