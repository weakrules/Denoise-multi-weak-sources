import torch
import numpy as np

def data_load(data_file, n_high_cov):
    data_dict = torch.load(data_file)
    unlabeled = data_dict['unlabeled']
    labeled = data_dict['labeled']
    validation = data_dict['validation']
    test = data_dict['test']

    x_lf_u = unlabeled['lf'].float()
    x_lf_l = labeled['lf'].float()
    x_lf_v = validation['lf'].float()
    x_lf_t = test['lf'].float()

    x_lf_training = (torch.cat((x_lf_u, x_lf_l), 0)).numpy()

    lf_count = np.zeros(len(x_lf_training))
    lf_number = x_lf_training.shape[1]  # how many LF we have
    for i in range(len(x_lf_training)):
        lf_count[i] = lf_number - np.count_nonzero(x_lf_training[i] == -1)

    high_coverage_ids = np.where(lf_count > n_high_cov)[0]  # select samples that have been labeled > n

    labeled_insts_idx = torch.tensor(high_coverage_ids)
    print("high_coverage_ids", len(high_coverage_ids))
    low_coverage_ids = list(set([i for i in range(len(x_lf_training))]) - set(high_coverage_ids))
    unlabeled_insts_idx = torch.tensor(low_coverage_ids)

    x_lf_training = torch.tensor(x_lf_training)
    x_lf_u = x_lf_training[unlabeled_insts_idx]  # unlabeled set, D_U
    x_lf_l = x_lf_training[labeled_insts_idx]  # labeled set, D_V

    lf_count_2 = np.zeros(len(x_lf_u))
    lf_number_2 = x_lf_u.shape[1]
    for i in range(len(x_lf_u)):
        lf_count_2[i] = lf_number_2 - np.count_nonzero(x_lf_u[i] == -1)
    no_cover_ids = np.where(lf_count_2 == 0)[0]
    print("no_cover_ids", len(no_cover_ids))

    x_u = unlabeled['bert_feature']
    x_l = labeled['bert_feature']
    x_training = torch.cat((x_u, x_l), 0)
    x_u = x_training[unlabeled_insts_idx]
    x_l = x_training[labeled_insts_idx]
    x_v = validation['bert_feature']
    x_t = test['bert_feature']

    y_u = unlabeled['major_label'].long()
    y_l = labeled['major_label'].long()
    y_training = torch.cat((y_u, y_l), 0)
    y_u = y_training[unlabeled_insts_idx]
    y_l = y_training[labeled_insts_idx]

    y_u_gt = unlabeled['label'].long()
    y_l_gt = labeled['label'].long()
    y_gt_training = torch.cat((y_u_gt, y_l_gt), 0)
    y_u_gt = y_gt_training[unlabeled_insts_idx]
    y_l_gt = y_gt_training[labeled_insts_idx]

    y_v = validation['label'].long()
    y_t = test['label'].long()

    training_set = (x_u, x_l, y_u, y_l, x_lf_u, x_lf_l, y_u_gt, y_l_gt)
    validation_set = (x_v, y_v, x_lf_v)
    test_set = (x_t, y_t, x_lf_t)

    n_features = x_u.size(1)
    #    n_rules = x_lf_u.size(1)
    n_rules = len(x_lf_training[0])
    return training_set, validation_set, test_set, n_features, n_rules, no_cover_ids, y_l


def save_log(log_prefix, train_metrics_list, val_metrics_list, test_metrics, lr, hidden, c2, c3, c4):
    log_name = log_prefix + '_{}_{}_{}_{}.log'.format(lr, hidden, c2, c3, c4)
    with open(log_name, 'w') as f:
        f.write('training supervised accuracy;\t\ttraining supervised loss;\t\t'
                'training supervised weight;\t\ttraining unsupervised accuracy;\t\t'
                'training unsupervised loss;\t\ttraining total loss\t\t'
                'validation supervised accuracy;\t\t validation attention accuracy\t\t'
                'validation ensemble accuracy;\n')
        for train_metrics, val_metrics in zip(train_metrics_list, val_metrics_list):
            f.write('{:4f}\t\t{:4f}\t\t{:4f}\t\t{:4f}\t\t{:4f}\t\t{:4f}\t\t'
                    '{:4f}\t\t{:4f}\t\t{:4f}\n'
                    .format(train_metrics[0], train_metrics[1], train_metrics[2],
                            train_metrics[3], train_metrics[4], train_metrics[5],
                            val_metrics[1], val_metrics[2], val_metrics[0])
                    )
        f.write('[test] accuracy: {:.4f}, attention accuracy: {:.4f},'
                ' ensemble accuracy: {:.4f}'.format(test_metrics[1], test_metrics[2], test_metrics[0])
                )