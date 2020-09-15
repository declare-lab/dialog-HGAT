import torch
import numpy as np
import copy

def pool_fn(h, mask, t='max'):
    if t == 'max':
        h = h.masked_fill(mask, -1e10)
        return torch.max(h, 1)[0]
    elif t == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    elif t == 'sum': return h.sum(1)


def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    Used for train word embeddings
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad


def acc_score(y_true, y_pred):
    total = 0
    for pred, label in zip(y_pred, y_true):
        total += np.sum(  (pred > 0.5) == (label > 0.5)  ) / len(pred)
    return total / len(y_pred)


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))


def f1_score(y_true, y_pred, T2=None):
    """T2 ranges from 0 to 0.5"""
    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        
        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return f_1, precision, recall


    label_ids = []
    for o in y_true:
        label_id = []
        assert(len(o) == 36)
        for i in range(36):
            if o[i] == 1:
                label_id += [i]
        if len(label_id) == 0:
            label_id = [36]
        label_ids += [label_id]
    assert len(label_ids) == len(y_pred)

    if T2:
        devp = getpred(y_pred, T2=T2)
        f_1, prec, rec = geteval(devp, label_ids)
        return f_1, T2, prec, rec, devp, label_ids 

    else:
        bestT2 = bestf_1 = 0
        bestPrec, bestRec = 0, 0
        best_devp, best_label = None, None
        for T2 in range(51):
            devp = getpred(y_pred, T2=T2/100.)
            f_1, prec, rec = geteval(devp, label_ids)
            if f_1 > bestf_1:
                bestf_1 = f_1
                bestT2 = T2/100.
                bestPrec, bestRec = prec, rec
                best_devp = copy.deepcopy(devp)
                best_label = copy.deepcopy(label_ids)

        return bestf_1, bestT2, bestPrec, bestRec, best_devp, best_label


def f1c_score(pred, data, T2=None):
    if T2:
        devp = getpred(pred, T1=0.5, T2=T2)
        prec, rec, f_1c = f1c(devp, data)
        return prec, rec, f_1c, T2

    else:
        bestT2c = bestf_1c = 0
        for T2 in range(51):
            devp = getpred(pred, T2=T2/100.)
            prec, rec, f_1c = f1c(devp, data)
            if f_1c > bestf_1c:
                bestf_1c, bestT2c = f_1c, T2/100.
                bestprec, bestrec = prec, rec
        return bestprec, bestrec, bestf_1c, bestT2c

           
           

def f1c(devp, data):
    index = 0
    precisions = []
    recalls = []
    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            correct_sys, all_sys = 0, 0
            correct_gt = 0
            
            x = data[i][1][j]["x"].lower().strip()
            y = data[i][1][j]["y"].lower().strip()
            t = {}
            for k in range(len(data[i][1][j]["rid"])):
                if data[i][1][j]["rid"][k] != 36:
                    t[data[i][1][j]["rid"][k]] = data[i][1][j]["t"][k].lower().strip()

            l = set(data[i][1][j]["rid"]) - set([36])

            ex, ey = False, False
            et = {}
            for r in range(36):
                et[r] = r not in l

            for k in range(len(data[i][0])):
                o = set(devp[index]) - set([36])
                e = set()
                if x in data[i][0][k].lower():
                    ex = True
                if y in data[i][0][k].lower():
                    ey = True
                if k == len(data[i][0])-1:
                    ex = ey = True
                    for r in range(36):
                        et[r] = True
                for r in range(36):
                    if r in t:
                        if t[r] != "" and t[r] in data[i][0][k].lower():
                            et[r] = True
                    if ex and ey and et[r]:
                        e.add(r)
                correct_sys += len(o & l & e)
                all_sys += len(o & e)
                correct_gt += len(l & e)
                index += 1

            precisions += [correct_sys/all_sys if all_sys != 0 else 1]
            recalls += [correct_sys/correct_gt if correct_gt != 0 else 0]

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1

def getpred(result, T1 = 0.5, T2 = 0.4):
    ret = []
    for i in range(len(result)):
        r = []
        maxl, maxj = -1, -1
        for j in range(len(result[i])):
            if result[i][j] > T1:
                r += [j]
            if result[i][j] > maxl:
                maxl = result[i][j]
                maxj = j
        if len(r) == 0:
            if maxl <= T2:
                r = [36]
            else:
                r += [maxj]
        ret += [r]
    return ret


