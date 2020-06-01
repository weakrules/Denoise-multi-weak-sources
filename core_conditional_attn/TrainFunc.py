import torch
import torch.nn.functional as F
import numpy as np


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch, model, optimizer, data, device, no_cover_ids, Z, z, n_class, y_l_new, c2, c3, c4, x0=100, k=0.01):
    x_u, x_l, y_u, y_l, x_lf_u, x_lf_l, y_u_gt, y_l_gt = map(lambda x: x.to(device), data)

    model.train()

    optimizer.zero_grad()
    y_l = y_l_new.to(device)
    Z = Z.to(device)
    predict_l, predict_u, lf_y_l, lf_y_u, fix_score = model(x_l, x_u, x_lf_l, x_lf_u)
    # print(fix_score)
    loss_sup = F.nll_loss(predict_l, y_l)

    # ---use temporal ensembling ideas, compare history output for uncovered ids
    alpha = 0.6
    quad_diff = 0
    if epoch > 10:
        zcomp = z.to(device)
        quad_diff = ((zcomp - predict_u[no_cover_ids]) ** 2).mean()
        outputs = predict_u[no_cover_ids].data.clone()
        Z = alpha * Z + (1. - alpha) * outputs
        z = Z * (1. / (1. - alpha ** (epoch + 1)))
        '''
        apply bert predictions to no cover ids for attention net, may delete
        '''
        lf_y_u[no_cover_ids] = torch.exp(predict_u[no_cover_ids])

    loss_sup_weight = F.nll_loss(lf_y_l, y_l)

    accu_sup = accuracy(predict_l, y_l_gt)
    accu_bert = accuracy(predict_l, y_l)
    accu_unsup = accuracy(lf_y_l, y_l_gt)

    # print("wrong in y_l", len(np.where((y_l.data != y_l_gt.data))[0]))

    w2 = c2 / (1 + np.exp(-k * (epoch - x0)))
    w3 = c3 / (1 + np.exp(-k * (epoch - x0)))
    w4 = c4 / (1 + np.exp(-k * (epoch - x0)))
    w1 = 1 - w2 - w3 - w4

    consistency_loss = w3 * ((predict_l - lf_y_l) ** 2).mean()
    self_learning_loss =  w4 * quad_diff 

    loss = w1 * loss_sup + w2 * loss_sup_weight + consistency_loss + self_learning_loss

    loss.backward()
    optimizer.step()

    metrics = [accu_sup, loss_sup, loss_sup_weight, accu_unsup, consistency_loss, loss]

    # --- update y_l, pseduo-clean labels---
    lf_y_l_new = torch.zeros((x_lf_l.size(0), n_class), dtype=torch.float, device=x_l.device)
    if epoch >= 0:
        for k in range(n_class):
            lf_y_l_new[:, k] = (fix_score.unsqueeze(0).repeat([x_lf_l.size(0), 1]) * (x_lf_l == k).float()).sum(dim=1)
        lf_y_l_new /= torch.sum(lf_y_l_new, dim=1).unsqueeze(1)
        lf_y_l_new[lf_y_l_new != lf_y_l_new] = 0  # handle the 'nan' (divided by 0) problem
        lf_y_l_new = F.log_softmax(lf_y_l_new, dim=1).detach()
        pred_y_l_new = lf_y_l_new.max(1)[1]
    else:
        pred_y_l_new = y_l
    return fix_score, metrics, pred_y_l_new, z


def test(model, data, fix_score, device, n_class):
    x_t, y_t, x_lf_t = map(lambda x: x.to(device), data)

    model.eval()
    with torch.no_grad():
        output_text = model.fc_model(x_t)
        pred_text = output_text.max(1)[1]

        # --- return the attention module soft predictions ---
        output_y, _ = model.attention(x_lf_t, x_t)
        pred_y = output_y.max(1)[1]

        # --- accuracy for each module ---

        accu_fc = accuracy(output_text, y_t)
        accu_attn = accuracy(output_y, y_t)

        # --- find the best accuracy between the two predictions---
        pred_text = pred_text.cpu().numpy()
        pred_y = pred_y.cpu().numpy()
        pred = pred_y
        count_not_same = 0
        for i in range(len(pred_text)):
            if pred_text[i] != pred_y[i]:
                count_not_same = count_not_same + 1
                preda = pred_text[i]
                predb = pred_y[i]
                if output_text[i][preda] > output_y[i][predb] * 1.1:
                    pred[i] = pred_text[i]
                else:
                    pred[i] = pred_y[i]
        count = 0
        for i in range(len(pred)):
            if pred[i] == y_t[i]:
                count = count + 1
        accu = count / len(pred)

    return accu, accu_fc, accu_attn
