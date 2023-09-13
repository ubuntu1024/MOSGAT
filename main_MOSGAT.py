""" Example for MOSGAT classification
"""
from train_test import train_test


if __name__ == "__main__":    
    data_folder = 'BRCA'
    testonly = True
    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 1000
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-4
    
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    
train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, testonly,
               num_epoch_pretrain, num_epoch)