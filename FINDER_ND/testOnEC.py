#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 这个是我实验的内容
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm
import networkx as nx
import EC

def main():
    dqn = FINDER()
    data_test_path = 'data/synthetic/uniform_cost/'
    # data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    data_test_name = ['30-50', '50-100']
    model_file = 'models/nrange_30_50_iter_78000.ckpt'
    file_path  = 'results/FINDER_ND/synthetic'
    
    if not os.path.exists('results/FINDER_ND'):
        os.mkdir('results/FINDER_ND')
    if not os.path.exists('results/FINDER_ND/synthetic'):
        os.mkdir('results/FINDER_ND/synthetic')
        
    with open('%s/result.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            dqn.getGraphAndSol(data_test, model_file)

            # print(a.shape)
            # ec_g = EC.ECGraph(g.size(),a)
            # print(ec_g.Lambda)
            # ec = EC.EC(a, ec_g.Lambda * 0.5)
            # s = ec.HDG()

            # print(sol)
            # xx, Lambda = ec.iterativesecure(sol)
            # print(xx)
            # cost = 0
            # for k in xx:
            #     if k == 1 or k == 1.0:
            #         cost = cost + 1
            # print("hdg cost: ",s.Cost," FINDER cost: ",cost)

            fout.write('%.2f+-%.2f,' % (1 * 100, 2 * 100))
            fout.flush()
            print('\ndata_test_%s has been tested!' % data_test_name[i])

if __name__=="__main__":
    main()