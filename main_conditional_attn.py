import argparse

import torch.optim as optim
import torch
import numpy as np
import os
import copy
from tqdm.auto import tqdm
from core_conditional_attn.Model import AssembleModel
from core_conditional_attn.TrainFunc import train, test
from core_conditional_attn.Data import data_load, save_log

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return None


def parse_args():
    file_name = '.'.join(os.path.basename(__file__).split('.')[:-1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_file', type=str, default=None,
                        help='full dataset location.')
    parser.add_argument('--ds', type=str, required=True,
                        choices=['youtube', 'imdb', 'yelp', 'agnews', 'spouse'],
                        help='dataset indicator.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='ban cuda devices.')
    parser.add_argument('--fast_mode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=128,
                        help='network hidden size.')
    parser.add_argument('--c2', type=float, default=0.1)
    parser.add_argument('--c3', type=float, default=0.1)
    parser.add_argument('--c4', type=float, default=0.1)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--x0', type=int, default=50)
    parser.add_argument('--unlabeled_ratio', type=float, default=0.8)
    parser.add_argument('--log_prefix', type=str, default=os.path.join(
        'log_files', file_name
    ))
    parser.add_argument('--ft_log', type=str, default=os.path.join(
        'ft_logs', file_name
    ))
    parser.add_argument('--n_high_cov', type=int, default=1)
    args = parser.parse_args()

    # manipulate arguments
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.pt_file:
        args.pt_file = 'dataset/{}/{}_organized_nb.pt'.format(args.ds, args.ds)
    if args.log_prefix is 'None':
        args.log_prefix = None
    else:
        args.log_prefix += '_{}.log'.format(args.ds)
    args.ft_log += '_{}_new.log'.format(args.ds)
    args.num_class = 2

    print(args)

    return args


def main():
    args = parse_args()

    if args.cuda:
        print('Using GPU!')
    device = torch.device("cuda" if args.cuda else "cpu")
    set_seed_everywhere(args.seed, args.cuda)

    # load data
    training_set, validation_set, testing_set, \
        n_features, n_rules, no_cover_ids, y_l = data_load(args.pt_file, args.n_high_cov) #TODO: change trans/in
    print('Data loaded from pt file!')

    print(' -------- ')
    print('lr: {}, hidden: {}, c2: {}, c3: {}'.format(args.lr, args.hidden, args.c2, args.c3))
    model = AssembleModel(
        n_features=n_features,
        n_rules=n_rules,
        d_hid=args.hidden,
        n_class=args.num_class
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_accu = 0
    best_model_wts = None
    best_fix_score = None

    train_metrics_list = list()
    val_metrics_list = list()

    print('training model...')
    Z = torch.zeros(len(no_cover_ids), args.num_class).float()  # intermediate values
    z = torch.zeros(len(no_cover_ids), args.num_class).float()  # temporal outputs
    # outputs = torch.zeros(len(no_cover_ids), args.num_class).float()  # current outputs
    pred_y_l_new = y_l.clone()
    for epoch in tqdm(range(args.epoch)):
        fix_score, train_metrics, pred_y_l_new, z = train(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            data=training_set,
            device=device,
            no_cover_ids=no_cover_ids,
            Z=Z,
            z=z,
            n_class=args.num_class,
            y_l_new=pred_y_l_new,
            k=args.k,
            x0=args.x0,
            c2=args.c2,
            c3=args.c3,
            c4=args.c4
        )
        train_metrics_list.append(train_metrics)

        val_accu, val_accu_fc, val_accu_attn = test(
            model, validation_set, fix_score, device, n_class=args.num_class
        )
        val_metrics_list.append([val_accu, val_accu_fc, val_accu_attn])

        if val_accu >= best_accu:
            best_accu = val_accu
            best_fix_score = fix_score
            best_model_wts = copy.deepcopy(model.state_dict())

    print('[validation] best accuracy: {}'.format(best_accu))
    model.load_state_dict(best_model_wts)
    test_accu, test_accu_fc, test_accu_attn = test(
        model, testing_set, best_fix_score, device, n_class=args.num_class
    )
    test_metrics = (test_accu, test_accu_fc, test_accu_attn)
    print('[test] accuracy: {:.4f}, attention accuracy: {:.4f},'
          ' ensemble accuracy: {:.4f}'.format(test_accu_fc, test_accu_attn, test_accu))

    if args.log_prefix:
        save_log(args.log_prefix, train_metrics_list, val_metrics_list, test_metrics,
                 args.lr, args.hidden, args.c2, args.c3, args.c4)

    with open(args.ft_log, 'a', encoding='utf-8') as f:
        f.write('lr, {}, hidden, {}, c2, {}, c3, {}, c4, {}, n_high, {} '
                'validation_accu, {:.4f}, test_accu, {:.4f}\n'
                .format(args.lr, args.hidden, args.c2, args.c3, args.c4, args.n_high_cov, best_accu, test_accu)
                )

    file_name = '.'.join(os.path.basename(__file__).split('.')[:-1])
    torch.save({
        'state_dict': model.state_dict,
        'fix_score': best_fix_score
    }, 'model/model_{}_{}_{}_{}_{}_{}_{}.cpk'.format(
        file_name, args.ds, args.lr, args.hidden, args.c2, args.c3, args.c4
    ))


if __name__ == '__main__':
    main()
