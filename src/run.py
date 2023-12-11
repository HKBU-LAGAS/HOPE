from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import random
import math
import os
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import time
from scipy.sparse import dok_matrix, csc_matrix, csr_matrix, hstack, vstack, dia_matrix
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from scipy.linalg import qr
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh #Find k eigenvalues and eigenvectors of the real symmetric square matrix or complex Hermitian matrix A
from scipy.sparse.linalg import eigs  #Find k eigenvalues and eigenvectors of the square matrix A
from sklearn.utils.extmath import randomized_svd
from scipy.special import softmax

# https://scikit-learn.org/stable/modules/clustering.html
from sklearn.cluster import AffinityPropagation, KMeans, SpectralClustering, DBSCAN, MeanShift, Birch

from sklearn.cluster._spectral import discretize

from scipy.sparse.linalg import inv

from scipy.linalg import qr, svd
from scipy.sparse.linalg import svds

from rounding import FNEM_rounding, SNEM_rounding

np.get_include()

def output(args, labels):
    if not os.path.exists('../cluster/'+args.data+'/'):
        os.makedirs('../cluster/'+args.data+'/')

    with open('../cluster/'+args.data+'/'+args.algo+'.txt', 'w') as f:
    	for i in range(len(labels)):
    		f.write(str(i)+'\t'+str(labels[i])+'\n')


def load_matrix(data):
	filepath = '../datasets/'+data+'/graph.txt'
	print('loading '+filepath)

	IJV = np.fromfile(filepath,sep="\t").reshape(-1,3)

	row = IJV[:,0].astype(np.int)
	col = IJV[:,1].astype(np.int)
	data = IJV[:,2]

	X = csr_matrix( (data,(row,col)) )
	print("Data size:", X.shape)

	return X


def BGC(args):
    W = load_matrix(args.data)

    c = np.array(np.sqrt(W.sum(axis=0)))
    c[c==0]=1
    c = 1.0/c
    c = c.flatten().tolist()
    cinv = diags(c)

    F = preprocessing.normalize(W, norm='l1', axis=1)
    B = W.T
    
    Bc = cinv.dot(B)
    
    start = time.time()

    n = W.shape[1]
    dim=int(args.dim*args.k)

    print("dimension=%d"%(dim))

    r = np.array(np.sqrt(W.sum(axis=1)))
    r[r==0]=1
    r = 1.0/r
    r = diags(r.flatten().tolist())
    L= Bc.dot(r)
    U, s, V = randomized_svd(L, n_components=dim, n_iter=20)
    s = s**2

    alpha = args.alpha
    s = (1.0-alpha)/(1.0-alpha*(s))
    s = np.array(s).flatten().tolist()
    s = diags(s).todense()
    U = U.dot(s)
    U = F.dot(U)
    U = preprocessing.normalize(U, norm='l2', axis=1)

    print("start performing k-means...")
    # export OPENBLAS_NUM_THREADS=2
    # export OMP_NUM_THREADS=2 
    clustering = KMeans(n_clusters=args.k, random_state=1024).fit(U)
    labels = clustering.labels_


    elapsedTime = time.time()-start
    print("Elapsed time (secs) for %s clustering: %f"%(args.algo, elapsedTime))

    output(args, labels)


def SNEM(args):
    W = load_matrix(args.data)

    c = np.array(np.sqrt(W.sum(axis=0)))
    c[c==0]=1
    c = 1.0/c
    c = diags(c.flatten().tolist())

    P = preprocessing.normalize(W, norm='l1', axis=1)
    R = c.dot(W.T)
    r = np.array(np.sqrt(W.sum(axis=1)))
    r[r==0]=1
    r = 1.0/r
    r = diags(r.flatten().tolist())
    R= R.dot(r)
    

    start = time.time()
    dim=int(args.dim*args.k)

    print("dimension=%d"%(dim))

    U, s, V = randomized_svd(R, n_components=dim, n_iter=5)
    s = s**2

    alpha = args.alpha
    s = (1.0-alpha)/(1.0-alpha*(s))
    s = np.array(s).flatten().tolist()
    s = diags(s).todense()
    U = U.dot(s)
    U = P.dot(U)
    U = preprocessing.normalize(U, norm='l2', axis=1)

    U, s, V = randomized_svd(U, n_components=args.k, n_iter=5)

    labels = SNEM_rounding(U, 40)

    elapsedTime = time.time()-start
    print("Elapsed time (secs) for %s clustering: %f"%(args.algo, elapsedTime))

    output(args, labels)


def FNEM(args):
    W = load_matrix(args.data)

    c = np.array(np.sqrt(W.sum(axis=0)))
    c[c==0]=1
    c = 1.0/c
    c = diags(c.flatten().tolist())

    P = preprocessing.normalize(W, norm='l1', axis=1)
    R = c.dot(W.T)
    r = np.array(np.sqrt(W.sum(axis=1)))
    r[r==0]=1
    r = 1.0/r
    r = diags(r.flatten().tolist())
    R= R.dot(r)
    

    start = time.time()
    dim=int(args.dim*args.k)

    print("dimension=%d"%(dim))

    U, s, V = randomized_svd(R, n_components=dim, n_iter=5)
    s = s**2

    alpha = args.alpha
    s = (1.0-alpha)/(1.0-alpha*(s))
    s = np.array(s).flatten().tolist()
    s = diags(s).todense()
    U = U.dot(s)
    U = P.dot(U)
    U = preprocessing.normalize(U, norm='l2', axis=1)

    U, s, V = randomized_svd(U, n_components=args.k, n_iter=5)

    labels = FNEM_rounding(U, 100)

    elapsedTime = time.time()-start
    print("Elapsed time (secs) for %s clustering: %f"%(args.algo, elapsedTime))

    output(args, labels)


def main():
    parser = ArgumentParser("Our",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='default',
                        help='data name.')

    parser.add_argument('--k', default=10, type=int,
                        help='#clusters.')

    parser.add_argument('--dim', default=5, type=float,
                        help='dim.')

    parser.add_argument('--alpha', default=0.3, type=float,
                        help='alpha')

    parser.add_argument('--algo', default="BGC",
                        help='method name.')


    args = parser.parse_args()
    print(args)
    
    DATA2K={'cora': 7, 'citeseer': 6, 'blogcatalog': 6, 'flickr': 9, 'pubmed': 3, 'corafull': 70, 'asia_lastfm': 18, 'lastfm': 239, 'mind': 18, 'mag': 8}
    if args.data in DATA2K:
        args.k = DATA2K[args.data]
    
    print("data=%s, #clusters=%d"%(args.data, args.k))

    if args.algo=="BGC":
        BGC(args)
    elif args.algo=="SNEM":
        SNEM(args)
    elif args.algo=="FNEM":
        FNEM(args)
    else:
        print("Unknown Algorithm!!!")

if __name__ == "__main__":
    sys.exit(main())
