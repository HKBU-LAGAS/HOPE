#from sklearn.metrics.cluster import normalized_mutual_info_score
#from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from clu_metrics import normalized_mutual_info_score
from clu_metrics import adjusted_rand_score
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import random
import math
import os
import numpy as np
from munkres import Munkres

def load_labels(fpath):
    print('loading '+fpath)
    IJ = np.fromfile(fpath,sep="\t").reshape(-1,2)
    row = IJ[:,0].astype(np.int)
    col = IJ[:,1].astype(np.int)
    
    n = max(row)+1
    print("%d lines loaded"%n)
    
    labels=np.zeros(n, dtype=int)  #[0]*n
    
    for i in range(len(row)):
        labels[row[i]]=col[i]
    
    return labels


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return 0, 0, 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)

    f1_macro = f1_score(y_true, new_predict, average="macro")
    f1_micro = f1_score(y_true, new_predict, average="micro")

    #auc = roc_auc_score(y_true, new_predict, multi_class='ovr')

    return acc, f1_macro, f1_micro #, auc


def eval(args):
    true_label_fpath = '../datasets/'+args.data+'/labels.txt'
    
    true_labels = load_labels(true_label_fpath)
    
    pred_label_fpath = '../cluster/'+args.data+'/'+args.algo+'.txt'
    
    pred_labels = load_labels(pred_label_fpath)

    if len(true_labels)!=len(pred_labels):
        pred_labels=pred_labels[0:len(true_labels)]

    #print(len(set(pred_labels)))
    #print(len(set(true_labels)))
    
    #NMI = normalized_mutual_info_score(true_labels, pred_labels)
    #ARI = adjusted_rand_score(true_labels, pred_labels)
    
    #print("%s Method's NMI: %f"%(args.algo, NMI))
    #print("%s Method's ARI: %f"%(args.algo, ARI))
    
    ACC = 0
    f1_macro = 0
    f1_micro = 0
    ACC, f1_macro, f1_micro = cluster_acc(np.array(true_labels), np.array(pred_labels))
    
    # print(len(set(true_labels)), len(set(pred_labels)))
    print("%s Method's ACC: %.3f"%(args.algo, ACC))
    print("%s Method's F1-Macro: %.3f"%(args.algo, f1_macro))
    #print("%s Method's F1-Micro: %.3f"%(args.algo, f1_micro))
    #print("%s Method's AUC: %.3f"%(args.algo, auc))

    
    NMI = normalized_mutual_info_score(true_labels, pred_labels)
    ARI = adjusted_rand_score(true_labels, pred_labels)
    
    print("%s Method's NMI: %.3f"%(args.algo, NMI))
    print("%s Method's ARI: %.3f"%(args.algo, ARI))
    
    if len(true_labels)>10000000000000000:
        num_sample=1000
        NMI=0
        ARI=0
        for i in range(num_sample):
            randomlist = random.sample(range(len(true_labels)), 10000)
            NMI_tmp = normalized_mutual_info_score(true_labels[randomlist], pred_labels[randomlist])
            ARI_tmp = adjusted_rand_score(true_labels[randomlist], pred_labels[randomlist])
            NMI+=NMI_tmp
            ARI+=ARI_tmp

        NMI = NMI*1.0/num_sample
        ARI = ARI*1.0/num_sample

        print("%s Method's approx NMI: %.3f"%(args.algo, NMI*1.0/num_sample))
        print("%s Method's approx ARI: %.3f"%(args.algo, ARI*1.0/num_sample))

    print("OVERALL: %.3f;%.3f;%.3f;%.3f"%(ACC, f1_macro, NMI, ARI))

def main():
    parser = ArgumentParser("Our",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='default',
                        help='data name.')

    parser.add_argument('--algo', default='LE',
                        help='method name.')


    args = parser.parse_args()
    
    eval(args)

if __name__ == "__main__":
    sys.exit(main())
