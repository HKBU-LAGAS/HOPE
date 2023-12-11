# -*- coding: utf-8 -*-

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
# from sklearn.cluster.k_means_ import k_means
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix

from sklearn import preprocessing


def FNEM_rounding(vectors, T=100):
    vectors = as_float_array(vectors)

    n_samples = vectors.shape[0]
    n_feats = vectors.shape[1]

    labels = vectors.argmax(axis=1)
    print(type(labels), labels.shape)
    vectors_discrete = csc_matrix(
            (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
            shape=(n_samples, n_feats))

    vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
    vectors_sum[vectors_sum==0]=1
    vectors_discrete = vectors_discrete*1.0/vectors_sum

    for _ in range(T):
        t_svd = vectors.T.dot(vectors_discrete)
        U, S, Vh = np.linalg.svd(t_svd)
        Q = np.dot(U, Vh)
        vectors = vectors.dot(Q)

        # S.sum()

        vectors = as_float_array(vectors)
        labels = vectors.argmax(axis=1)
        # print(type(labels), labels.shape)
        vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_feats))

        vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
        vectors_sum[vectors_sum==0]=1
        vectors_discrete = vectors_discrete*1.0/vectors_sum

    return labels

def SNEM_rounding(vectors, T=100):
    vectors = as_float_array(vectors)
    n_samples = vectors.shape[0]
    n_feats = vectors.shape[1]

    labels = vectors.argmax(axis=1)
    print(type(labels), labels.shape)
    vectors_discrete = csc_matrix(
            (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
            shape=(n_samples, n_feats))

    vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
    vectors_sum[vectors_sum==0]=1
    vectors_discrete = vectors_discrete*1.0/vectors_sum
    #vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)

    for _ in range(T):
        Q = vectors.T.dot(vectors_discrete)

        vectors_discrete = vectors.dot(Q)
        vectors_discrete = as_float_array(vectors_discrete)

        labels = vectors_discrete.argmax(axis=1)
        vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_feats))

        
        #vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)

        vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
        vectors_sum[vectors_sum==0]=1
        vectors_discrete = vectors_discrete*1.0/vectors_sum

    return labels


