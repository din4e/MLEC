#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EC的实验的内容
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm
import networkx as nx
import EC

def main():
    dqn = FINDER()
    
    data_test_path = 'data/synthetic/'
    data_list = ['degree_cost','uniform_cost','random_cost']
    data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    
    model_file = 'models/nrange_30_50_iter_78000.ckpt'
    file_path  = 'results/FINDER_ND/synthetic'
    
    if not os.path.exists('results/FINDER_ND'):
        os.mkdir('results/')
    if not os.path.exists('results/FINDER_ND/synthetic'):
        os.mkdir('results/synthetic')

    # fd = open( "f.txt", 'xt' )
    for data in data_list:
        with open('%s/result_1_%s.csv'%(file_path,data), 'w') as fout:
            for i in range(len(data_test_name)):
                data_test = data_test_path + data+'/' +data_test_name[i]
                le, _, ge , ans= dqn.getGraphAndSol(data_test, model_file)
                print("%s better case %.2f worse case %.2f"%(data_test_name[i],1.0*ge/100,1.0*le/100))
                # fd.write("%s better case %.2f worse case %.2f\n"%(data_test_name[i],1.0*ge/100,1.0*le/100))
                for line in ans:
                    fout.write("%d,%d,\n"%(line[0],line[1]))
                fout.flush()

if __name__=="__main__":
    main()