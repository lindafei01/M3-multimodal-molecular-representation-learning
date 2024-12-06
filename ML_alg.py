from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: main.py
@time: 2023/8/13 20:05
@desc:
'''
import os
import copy
import torch
import json
import logging
import warnings

import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from parser_args import get_args
from tensorboardX import SummaryWriter
from chemprop.data import StandardScaler
from utils.dataset import Seq2seqDataset, get_data, split_data, MoleculeDataset, InMemoryDataset, load_npz_to_data_list
from utils.evaluate import eval_rocauc, eval_rmse
from torch.utils.data import BatchSampler, RandomSampler, DataLoader
from build_vocab import WordVocab
from chemprop.nn_utils import NoamLR
from chemprop.features import mol2graph, get_atom_fdim, get_bond_fdim
from chemprop.data.utils import get_class_sizes
from models_lib.multi_modal import Multi_modal
from featurizers.gem_featurizer import GeoPredTransformFn
from torch.nn.utils import clip_grad_norm_
from visual_utils.visualization_module import YourVisualizationClass
from utils.weight import weighted_mse_loss
from utils.split import Splitter
from utils.dataset import get_unimol_data_config
import yaml
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data as geoData
from torch_geometric.data import Dataset as geoDataset
from torch_geometric.data import DataLoader as geoDataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
warnings.filterwarnings('ignore')

OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

class AgileDataset(geoDataset):
    def __init__(self, smiles, targets, task="regression"):
        super(geoDataset, self).__init__()
        self.smiles_data, self.labels = smiles, targets
        self.task = task
        self.conversion = 1

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == "classification":
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        elif self.task == "regression":
            y = torch.tensor(
                self.labels[index] * self.conversion, dtype=torch.float
            ).view(1, -1)
        data = geoData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)
    
def NNDataset(data, label=None, weight=None):
    return TorchDataset(data, label, weight)

def NNDataLoader(dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):

    dataloader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last)
    return dataloader

class TorchDataset(Dataset):
    def __init__(self, data, label=None,weight=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))
        self.weight = weight if weight is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx],self.weight[idx]

    def __len__(self):
        return len(self.data)

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))


def load_smiles_to_dataset(data_path):
    """tbd"""
    data_list = []
    with open(data_path, 'r') as f:
        tmp_data_list = [line.strip() for line in f.readlines()]
        tmp_data_list = tmp_data_list[1:]
    data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list)
    return dataset


def prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, device, training=False, 
                 unimol_data=None, agile_data=None, unimol_collate_fn=None):
    edge_batch1, edge_batch2 = [], []
    geo_gen = geo_data.get_batch(idx)
    node_id_all = [geo_gen[0].batch, geo_gen[1].batch]
    for i in range(geo_gen[0].num_graphs):
        edge_batch1.append(torch.ones(geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
        edge_batch2.append(torch.ones(geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
    edge_id_all = [torch.cat(edge_batch1), torch.cat(edge_batch2)]
    # 2D data
    mol_batch = MoleculeDataset([gnn_data[i] for i in idx])
    smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
    gnn_batch = mol2graph(smiles_batch, args)
    batch_mask_seq, batch_mask_gnn = list(), list()
    for i, (smile, mol) in enumerate(zip(smiles_batch, mol_batch.mols())):
        batch_mask_seq.append(torch.ones(len(smile), dtype=torch.long).to(device) * i)
        batch_mask_gnn.append(torch.ones(mol.GetNumAtoms(), dtype=torch.long).to(device) * i)
    batch_mask_seq = torch.cat(batch_mask_seq)
    batch_mask_gnn = torch.cat(batch_mask_gnn)
    mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
    targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
    if args.LDS and training:
        weights = torch.Tensor([gnn_data[i].weight for i in idx]).to(device)
    else:
        weights = None

    if unimol_data:
        unimol_batch_feature = np.asarray(unimol_data["unimol_input"])[idx]
        unimol_batch_target = np.asarray(targets.cpu())
        unimol_batch_dataset = NNDataset(unimol_batch_feature, unimol_batch_target)
        unimol_batch_dataloader = NNDataLoader(
            dataset=unimol_batch_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=unimol_collate_fn,
            drop_last=False,
        )
        for i, data in enumerate(unimol_batch_dataloader):
            unimol_data_batch = data
        assert i == 0


    else: 
        unimol_data_batch = None
    
    if agile_data:
        agile_batch_dataset = AgileDataset(smiles_batch, np.asarray(target_batch).squeeze().tolist(), task="regression" if args.task_type=="reg" else "classification")
        agile_batch_dataloader = geoDataLoader(dataset=agile_batch_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                )
        for i, data in enumerate(agile_batch_dataloader):
            agile_data_batch = data.to(device)
        assert i == 0

    else:
        agile_data_batch = None

    return seq_data[idx], seq_mask[idx], batch_mask_seq, gnn_batch, features_batch, batch_mask_gnn, geo_gen, \
           node_id_all, edge_id_all, mask, targets, weights, unimol_data_batch, agile_data_batch


def train(args, model, optimizer, scheduler, train_idx_loader, seq_data, seq_mask, gnn_data, geo_data, device, 
          epoch, training, unimol_data, agile_data):
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    if args.FDS:
        encodings = []
        labels1=[]
    for i, idx in enumerate(tqdm(train_idx_loader)):
        model.zero_grad()
        # 3D data
        seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, \
        gnn_batch_batch, geo_gen, node_id_all, \
        edge_id_all, mask, targets, weights, \
        unimol_data_batch, agile_data_batch = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, 
                                                           device, training=True, unimol_data=unimol_data, 
                                                           agile_data=agile_data, 
                                                           unimol_collate_fn=model.compound_encoder.batch_collate_fn if args.unimol else None)
        x_list, preds, repr1 = model(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
                              geo_gen, node_id_all, edge_id_all, targets, epoch, training, 
                              unimol_data_batch, agile_data_batch)
        if args.FDS:
            encodings.extend(repr1.data.cpu().numpy())
            labels1.extend(targets.data.squeeze(1).cpu().numpy())
            
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, targets, mask, weights)
        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        all_loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    if args.FDS:
        encodings, labels1 = torch.from_numpy(np.vstack(encodings)).cuda(0), \
                        torch.from_numpy(np.hstack(labels1)).cuda(0)
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(encodings, labels1, epoch)
    return all_loss.item(), lab_loss.item(), cl_loss.item(), model


@torch.no_grad()
def val(args, model, scaler, val_idx_loader, seq_data, seq_mask, gnn_data, geo_data, device, unimol_data=None, agile_data=None):
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    y_true = []
    y_pred = []
    for idx in val_idx_loader:
        # 3D data
        seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch, geo_gen, node_id_all, \
        edge_id_all, mask, targets, _, unimol_data_batch, agile_data_batch = prepare_data(args, idx, 
                                                                                          seq_data, seq_mask, 
                                                                                          gnn_data, geo_data, device,
                                                                                          training=False,
                                                                                          unimol_data=unimol_data, 
                                                                                          agile_data=agile_data,
                                                                                          unimol_collate_fn=model.compound_encoder.batch_collate_fn if args.unimol else None)
        x_list, preds, _ = model(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
                              geo_gen, node_id_all, edge_id_all, training=False, 
                              unimol_data_batch=unimol_data_batch, 
                              agile_data_batch=agile_data_batch)
        if scaler is not None and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu()).astype(np.float64)).to(device)
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, targets, mask, args.cl_loss)
        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        y_true.append(targets)
        y_pred.append(preds)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']
    # print('result:', result)
    return result, all_loss.item(), lab_loss.item(), cl_loss.item(), model, y_true, y_pred


@torch.no_grad()
def test(args, model, scaler, test_idx_loader, seq_data, seq_mask, gnn_data, geo_data, device, unimol_data=None, agile_data=None):
    y_true = []
    y_pred = []
    for idx in test_idx_loader:
        # 3D data

        seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, \
        features_batch, gnn_batch_batch, geo_gen, node_id_all, \
        edge_id_all, mask, targets, _, \
        unimol_data_batch, agile_data_batch= prepare_data(args, idx, seq_data, \
                                                          seq_mask, gnn_data, geo_data, \
                                                          device, training=False,
                                                        unimol_data=unimol_data, 
                                                        agile_data=agile_data,
                                                        unimol_collate_fn=model.compound_encoder.batch_collate_fn if args.unimol else None)
        x_list, preds, _ = model(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, 
                              features_batch, gnn_batch_batch,
                              geo_gen, node_id_all, edge_id_all, training=False, 
                              unimol_data_batch=unimol_data_batch, agile_data_batch=agile_data_batch)
        if scaler is not None and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu()).astype(np.float64))
        y_true.append(targets)
        y_pred.append(preds)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']
    # print('result:', result)
    return result, y_true, y_pred


def main(args):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import xgboost as xgb

    logs_file1 = os.path.join("LOG", args.dataset, f"rf{'_preprocess' if args.preprocess else ''}")
    if not os.path.exists(logs_file1):
        os.makedirs(logs_file1)

    # gnn data
    idx_data_path = 'data/lnp_unique.csv'
    # data_3d = load_smiles_to_dataset(args.data_path_3d)
    datas, args.seq_len = get_data(path=idx_data_path, args=args)
    # datas = MoleculeDataset(datas[0:8])
    args.output_dim = args.num_tasks = datas.num_tasks()
    args.gnn_atom_dim = get_atom_fdim(args)
    args.gnn_bond_dim = get_bond_fdim(args) + (not args.atom_messages) * args.gnn_atom_dim
    args.features_size = datas.features_size()
    # data split
    train_data, val_data, test_data = split_data(data=datas, split_type=args.split_type, sizes=args.split_sizes,
                                                 seed=args.seed, args=args)
    train_idx = [data.idx for data in train_data]
    val_idx = [data.idx for data in val_data]
    test_idx = [data.idx for data in test_data]

    train_idx.extend(val_idx)

    df = pd.read_csv(f'data/{args.dataset}.csv')

    start_feat_column = 'desc_ABC/10'
    feat_columns = df.columns[df.columns.get_loc(start_feat_column):]

    X = df[feat_columns].values
    Y = df["TARGET"].values

    if args.preprocess:
        from sklearn.feature_selection import VarianceThreshold
        from scipy.stats import pearsonr

        # 1. 移除低方差特征
        vt = VarianceThreshold(threshold=0.01)
        X_filtered = vt.fit_transform(X)

        # 2. 移除高度相关的特征
        corr_matrix = np.corrcoef(X_filtered.T)

        to_drop = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                corr, _ = pearsonr(X_filtered[:, i], X_filtered[:, j])
                if abs(corr) > 0.9:
                    to_drop.add(j)

        X = np.delete(X_filtered, list(to_drop), axis=1)



    X_train = X[train_idx]
    y_train = Y[train_idx]
    X_test = X[test_idx]
    y_test = Y[test_idx]


    # 创建随机森林回归模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # 拟合模型
    rf.fit(X_train, y_train)

    # 进行预测
    y_pred = rf.predict(X_test)

    if not os.path.exists(os.path.join(logs_file1, "result")):
        os.makedirs(os.path.join(logs_file1, "result"))

    with open(os.path.join(logs_file1, "result", "test_result.json"), "w") as file:
        json.dump({"mse": mean_squared_error(y_test, y_pred), "y_true": y_test.tolist(), "y_pred": y_pred.tolist()}, file)
    visualization_instance = YourVisualizationClass()
    visualization_instance._eval_stratified_classes(labels=y_test.flatten(), 
                                                                predictions=y_pred.flatten(), 
                                                                save_dir=os.path.join(logs_file1, "result"))
    

    with open(os.path.join(logs_file1, 'rf.pkl'), 'wb') as file:
        pickle.dump(rf, file)
    
#--------------------------------------------------------------------------------------------------------------------
    logs_file2 = os.path.join("LOG", args.dataset, f"xgb{'_preprocess' if args.preprocess else ''}")
    if not os.path.exists(logs_file2):
        os.makedirs(logs_file2)
    xgbmodel = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.1, n_estimators=100)
    xgbmodel.fit(X_train, y_train)
    y_pred = xgbmodel.predict(X_test)

    if not os.path.exists(os.path.join(logs_file2, "result")):
        os.makedirs(os.path.join(logs_file2, "result"))

    with open(os.path.join(logs_file2, "result", "test_result.json"), "w") as file:
        json.dump({"mse": mean_squared_error(y_test, y_pred), "y_true": y_test.tolist(), "y_pred": y_pred.tolist()}, file)
    visualization_instance = YourVisualizationClass()
    visualization_instance._eval_stratified_classes(labels=y_test.flatten(), 
                                                                predictions=y_pred.flatten(), 
                                                                save_dir=os.path.join(logs_file2, "result"))
    
    with open(os.path.join(logs_file2, 'xgb.pkl'), 'wb') as file:
        pickle.dump(xgbmodel, file)

#--------------------------------------------------------------------------------------------------------------------
    import lightgbm as lgb

    logs_file3 = os.path.join("LOG", args.dataset, f"lgb{'_preprocess' if args.preprocess else ''}")
    if not os.path.exists(logs_file3):
        os.makedirs(logs_file3)

    # 创建LightGBM回归模型
    lgb_model = lgb.LGBMRegressor(objective='regression',
                                num_leaves=31,
                                learning_rate=0.05,
                                n_estimators=100)

    # 拟合模型
    lgb_model.fit(X_train, y_train)

    # 进行预测
    y_pred = lgb_model.predict(X_test)

    if not os.path.exists(os.path.join(logs_file3, "result")):
        os.makedirs(os.path.join(logs_file3, "result"))

    with open(os.path.join(logs_file3, "result", "test_result.json"), "w") as file:
        json.dump({"mse": mean_squared_error(y_test, y_pred), "y_true": y_test.tolist(), "y_pred": y_pred.tolist()}, file)

    visualization_instance = YourVisualizationClass()
    visualization_instance._eval_stratified_classes(labels=y_test.flatten(), 
                                                predictions=y_pred.flatten(), 
                                                save_dir=os.path.join(logs_file3, "result"))

    with open(os.path.join(logs_file3, 'lgb.pkl'), 'wb') as file:
        pickle.dump(lgb_model, file)



if __name__ == "__main__":
    args = get_args()
    args.dataset = "lnp_unique_features"
    args.preprocess = True
    # args.ml_method = "RF"

    main(args)
