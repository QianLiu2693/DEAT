import multiprocessing

import gc

import numpy as np
import time
import torch
from scipy.spatial.distance import cdist
from log import logger

g = 1000000000

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set

def eval_alignment_by_sim_mat(embed1, embed2, alpha, m, top_k, nums_threads, csls=0, accurate=False,output = True):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, alpha, m, csls, nums_threads)
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            logger.info("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                       t_mrr,
                                                                                                       time.time() - t))
            
        else:
            logger.info("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
           
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, hits1

def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values

def CSLS_sim(sim_mat1, k, nums_threads):
    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values

def selu(x):
    alpha = 1.673263242354377284817042991635
    scale = 1.0507009821459666405
    return scale * np.where(x > 0.0, alpha * np.exp(x) - 1, x)

def sim_handler(embed1, embed2, alpha, m, k, nums_threads):
    
    alpha = alpha
    sim_mat_e = np.matmul(embed1, embed2.T)
    sim_mat_t = m
    
    sim_mat_e=1 / (1 + np.exp(-sim_mat_e))#sigmoid
    sim_mat_t=1 / (1 + np.exp(-sim_mat_t))#sigmoid

    alpha=0.3
    sim_mat = (1-alpha)*sim_mat_e + alpha*sim_mat_t
    #sim_mat = np.tanh(sim_mat)  #92.76
    #sim_mat = 1 / (1 + np.exp(-sim_mat))#sigmoid
    #sim_mat = np.maximum(0, sim_mat)#ReLU 92.6
    sim_mat =  np.maximum(0.01 * sim_mat, sim_mat)#leaky_relu 92.78
    #exps = np.exp(sim_mat - np.max(sim_mat))  # 防止溢出
    #sim_mat = exps / np.sum(exps)#Softmax
    #sim_mat = np.log(1 + np.exp(sim_mat))#Softplus
    #sim_mat = np.where(sim_mat > 0.0, sim_mat, 1.0 * (np.exp(sim_mat) - 1))#ELU92.66
    #sim_mat = selu(sim_mat) #92.7
    

    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat


