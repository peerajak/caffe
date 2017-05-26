import os
import sys
import scipy.io
import pymatbridge 
sys.path.insert(0, './python')
import numpy as np
import caffe
import pickle
import h5py
from pylab import *
import dt
import time
from datetime import timedelta

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('examples/PARSE_HogConvssvm26/parse_hogconvssvm26_manual_loss_solver.prototxt')
model_mat='/working3/peerajak/ChulaQE/Semister9/1_caffe2/examples/PARSE_HogConvssvm26/ExpParamsHOGssvm26_lmdb_batch5.mat'
ExpParamObj = scipy.io.loadmat(model_mat)
K = ExpParamObj.get('K').astype(np.int).squeeze() 
pa = ExpParamObj.get('pa').astype(np.int)
tolpart = int(ExpParamObj.get('tolpart'))
model = ExpParamObj.get('model') #get nontrained. You may get trained model by 'trained_model'
numFilters_tolpart= np.sum(K[0:tolpart])
#net_init = caffe.Net('examples/PARSE_HogConvssvm26/parse_hogconv_respmaps_deploy.prototxt', 'examples/PARSE_HogConvssvm26/save/parse_hogconvssvm26_trainStart_160610_152006_trainStop_160611_064739.caffemodel', caffe.TEST)
#for filter_i in range(numFilters_tolpart):
    #model['filters'][0,0][0,filter_i]['w'] =  solver.net.params['convssvm'][0].data[filter_i,:,:,:]
    #solver.net.params['convssvm'][0].data[filter_i,:,:,:] =np.transpose(model['filters'][0,0][0,filter_i]['w'],(2,0,1) ).astype(np.float64) 
#solver.net.params['convssvm'][0].data[...] = net_init.params['convssvm'][0].data
#solver.net.params['convssvm'][1].data[...] = net_init.params['convssvm'][1].data
#solver.net.params['conv1'][0].data[...] = net_init.params['conv1'][0].data
#solver.net.params['conv1'][1].data[...] = net_init.params['conv1'][1].data
starttime = time.strftime("%y%m%d_%H%M%S")
start_time = time.time()
#for i in range(2):
    #solver.step(1)
#   print '--- end iter ----'
#solver.solve()
#solver.step(1)
stoptime = time.strftime("%y%m%d_%H%M%S")
stop_time = time.time()
elapsed_time = stop_time - start_time
savename = 'examples/PARSE_HogConvssvm26/save/parse_hogconvssvm26_trainStart_' + starttime+'_trainStop_'+ stoptime+'.caffemodel'
hours, rem = divmod(stop_time-start_time, 3600)
minutes, seconds = divmod(rem, 60)
elaspedtime="{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),int(seconds))
print 'train starts ',starttime,' stops ',stoptime, 'total ',elaspedtime
solver.net.save(savename)
Experiment_name = 'parse_hogconvssvm26_trainStart_' + starttime+'_trainStop_'+ stoptime

# call matlab function to do detection on train/test images

#1. Save convssvm weights to .mat file, save the CNNfeat to .prototxt (net_feat_def) 
# and .caffemodel (savefeatname) then memorize the above 3 filenames 
caffe.set_mode_cpu()
net_feat_def = '/working3/peerajak/ChulaQE/Semister9/1_caffe2/examples/PARSE_HogConvssvm26/parse_hogconv_feat_deploy.prototxt'
net_feat = caffe.Net(net_feat_def, caffe.TEST) 
print net_feat.params['conv1'][0].data.shape #32,32,2,2
net_feat.params['conv1'][0].data[...] = solver.net.params['conv1'][0].data.copy()
savefeatname = '/working3/peerajak/ChulaQE/Semister9/1_caffe2/examples/PARSE_HogConvssvm26/save/parse_hogconvssvm26_trainStart_' + starttime+'_trainStop_'+ stoptime+'_feat.caffemodel'
net_feat.save(savefeatname)

savelosswname = '/working3/peerajak/ChulaQE/Semister9/1_caffe2/examples/PARSE_HogConvssvm26/save/parse_hogconvssvm26_trainStart_' + starttime+'_trainStop_'+ stoptime+'_lossw.mat'
lossw = solver.net.params['loss'][0].data.copy()
convssvm = solver.net.params['convssvm'][0].data.copy()
scipy.io.savemat(savelosswname, mdict={'lossw': lossw, 'convssvm':convssvm})

print savelosswname

#2. Call Matlab function with 3 filenames as parameters. Matlab function return the output dirname
#it saved the detection result, print the dir name out.

cmd_matlab = "matlab -nodisplay -nosplash -r \"res = detect_from_pycaffe_hogconvssvm_tolparts(\'"+net_feat_def+"\',\'"+savefeatname+"\',\'"+savelosswname+"\',\'"+model_mat+"\',\'"+Experiment_name+"\');quit\""

print cmd_matlab
#print 'net_feat_def='+net_feat_def
#print 'savefeatname='+savefeatname
#print 'savelosswname='+savelosswname
#print 'Experiment_name='+Experiment_name

#os.chdir('PARSE_createDb_and_CheckResults')
#os.system(cmd_matlab)


