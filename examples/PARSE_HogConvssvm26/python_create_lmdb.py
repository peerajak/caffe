import os
import sys
import scipy.io
sys.path.insert(0, './python')
import numpy as np
import caffe
import pickle
import lmdb
from pylab import *
import dt
import time
from datetime import timedelta
import re, fileinput, math


model_mat='/working3/peerajak/ChulaQE/Semister9/1_caffe/examples/PARSE_HogConvssvm26_PCP6598/ExpParamsHOGssvm26_lmdb_batch5.mat'
ExpParamObj = scipy.io.loadmat(model_mat)
K = ExpParamObj.get('K').astype(np.int).squeeze() 
pa = ExpParamObj.get('pa').astype(np.int)
experiment_param = ExpParamObj.get('experiment_param')
model = ExpParamObj.get('model')
numPos = int(ExpParamObj.get('numPos'))
numNeg = int(ExpParamObj.get('numNeg'))
numTest = int(ExpParamObj.get('numTest'))
num_train_batches = int(ExpParamObj.get('num_train_batches'))
num_test_batches = int(ExpParamObj.get('num_test_batches'))
numLevels = int(ExpParamObj.get('numLevels'))
batch_size = int(ExpParamObj.get('batch_size'))
tolpart = int(ExpParamObj.get('tolpart'))

    
train_batch_mat_path = 'examples/PARSE_HogConvssvm26_PCP6598/train_batch_mat'
from os import listdir
from os.path import isfile, join
matfiles = sorted([f for f in listdir(train_batch_mat_path) if isfile(join(train_batch_mat_path, f))])
num_train_batches = len(matfiles)

print('Writing labels')
print 'NumLevels = ',numLevels,'batch_size',batch_size
lmdb_data_name = 'examples/PARSE_HogConvssvm26_PCP6598/PARSE_traindata_lmdb'
lmdb_labels_name = 'examples/PARSE_HogConvssvm26_PCP6598/PARSE_trainlabels_lmdb'
total_impyra_counter_images = 0;
total_impyra_counter_labels = 0;
for file_i in matfiles:
    print train_batch_mat_path + '/' + file_i
    i_train_batch_Obj = scipy.io.loadmat(train_batch_mat_path + '/' + file_i)
    X = i_train_batch_Obj.get('batchdata').astype(np.float)
    y = i_train_batch_Obj.get('batchlabs').astype(np.float)
    print X.shape
    print('Writing image data')

    for idx in range(batch_size):
        in_db_data = lmdb.open(lmdb_data_name, map_size=int(1e12))
        with in_db_data.begin(write=True) as in_txn:
             for level_idx in range(numLevels):
                 im = X[:,:,:,idx*numLevels+level_idx].squeeze()
                 im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 1, 0)))         
                 in_txn.put('{:0>10d}'.format(total_impyra_counter_images), im_dat.SerializeToString()) 
                 total_impyra_counter_images = total_impyra_counter_images+1
                 #string_ = str(100*idx+in_idx+1) + ' / ' + str(len(Inputs))
                 #sys.stdout.write("\r%s" % string_)
                 #sys.stdout.flush()
        in_db_data.close()
    print('--written--')
    print('Writing labels')
    for idx in range(batch_size):
        in_db_data = lmdb.open(lmdb_labels_name, map_size=int(1e12))
        with in_db_data.begin(write=True) as in_txn:
             for level_idx in range(numLevels):
                 im = y[:,idx*numLevels+level_idx]
                 im_dat = caffe.io.array_to_datum(im.reshape(4*tolpart+12,1,1))
                 in_txn.put('{:0>10d}'.format(total_impyra_counter_labels), im_dat.SerializeToString())
                 total_impyra_counter_labels = total_impyra_counter_labels+1
                 #string_ = str(100*idx+in_idx+1) + ' / ' + str(len(Inputs))
                 #sys.stdout.write("\r%s" % string_)
                 #sys.stdout.flush()
        in_db_data.close()
    print('--written--')
    del X
    del y
    del i_train_batch_Obj 
