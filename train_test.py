""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter,Normal_value_of_sample,Samp_pro_tensor

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_TCP=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
            optim_dict["C{:}".format(i+1)].zero_grad()
            ci_loss = 0
            ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
            ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
            ci_loss.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_TCP:
        optim_dict["M"].zero_grad()
        out_feat = []
        c_loss = 0
        for i in range(num_view):
            feature=model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            out_feat.append(feature)
        c_loss, MMlogit = model_dict["Fus"](out_feat,label)
        c_loss.backward()
        optim_dict["M"].step()
        loss_dict["M"] = c_loss.detach().cpu().numpy().item()
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        code = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
        ci_list.append(code)
    clogit = model_dict["Fus"].infer(ci_list)
    clogit = clogit[te_idx,:]
    prob = F.softmax(clogit, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, testonly,
               num_epoch_pretrain, num_epoch):
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 3
        dim_he_list = [100,64,64]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400,300,200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    if testonly:
        if data_folder == 'ROSMAP':
            model_dict["E1"].load_state_dict(torch.load("./model/ROSMAP_E1.pth"))
            model_dict["E2"].load_state_dict(torch.load("./model/ROSMAP_E2.pth"))
            model_dict["E3"].load_state_dict(torch.load("./model/ROSMAP_E3.pth"))
            model_dict["C1"].load_state_dict(torch.load("./model/ROSMAP_C1.pth"))
            model_dict["C2"].load_state_dict(torch.load("./model/ROSMAP_C2.pth"))
            model_dict["C3"].load_state_dict(torch.load("./model/ROSMAP_C3.pth"))
            model_dict["Fus"].load_state_dict(torch.load("./model/ROSMAP_Fus.pth"))
        if data_folder == 'BRCA':
            model_dict["E1"].load_state_dict(torch.load("./model/BRCA_E1.pth"))
            model_dict["E2"].load_state_dict(torch.load("./model/BRCA_E2.pth"))
            model_dict["E3"].load_state_dict(torch.load("./model/BRCA_E3.pth"))
            model_dict["C1"].load_state_dict(torch.load("./model/BRCA_C1.pth"))
            model_dict["C2"].load_state_dict(torch.load("./model/BRCA_C2.pth"))
            model_dict["C3"].load_state_dict(torch.load("./model/BRCA_C3.pth"))
            model_dict["Fus"].load_state_dict(torch.load("./model/BRCA_Fus.pth"))
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        if num_class == 2:
            print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
        else:
            print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1 weighted: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Test F1 macro: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    else:
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        print("\nPretrain GATs...")
        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain)
        for epoch in range(num_epoch_pretrain):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_TCP=False)
        print("\nTraining...")
        optim_dict = init_optim(num_view, model_dict, lr_e)
        for epoch in range(num_epoch+1):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                print("\nTest: Epoch {:d}".format(epoch))
                if num_class == 2:
                    print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
                else:
                    print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                    print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
                    print()
