from __future__ import division
import caffe
import numpy as np
import scipy.io
import h5py
import dt
import mahotas

class LossAugmentedInfLossLayer(caffe.Layer):

    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        
       
        n = bottom[0].data.shape[0]
        model_mat='/working3/peerajak/ChulaQE/Semister9/1_caffe/examples/PARSE_ssvm26_lmdb_pretrained/ExpParamsHOGssvm26_lmdb_batch5.mat'
        ExpParamObj = scipy.io.loadmat(model_mat)      
        pretrained_modelObj = scipy.io.loadmat('/working3/peerajak/ChulaQE/Semister10/3_pose-release-ver1.2/code-full/pretrained_model_randdefs.mat')
        self.K = ExpParamObj.get('K').astype(np.int).squeeze() 
        self.pa = ExpParamObj.get('pa').astype(np.int).squeeze() 
        #print self.pa
        #self.model = ExpParamObj.get('model')
        self.model = pretrained_modelObj.get('pretrained_model')
        #self.model = ExpParamObj.get('modelpretrainedConvssvm32')
        self.wloss_len = int(self.model[0,0]['len'][0,0])
        self.blobs.add_blob(self.wloss_len)
        #print self.blobs[0].data.shape
        self.blobs[0].data[...] = self.model2vec_defbiasonly(self.model).copy()
        #self.svm_C = float(1.0) #SVM C should be added by setting lambda in solver.prototxt to 1/(2*C*m), m is total training samples
        self.firstRunFlag = True;
        self.dobj_dRespmaps = np.zeros(n,dtype='float64')
        self.grad_wloss = np.zeros(self.wloss_len,dtype='float64')
        self.numLevels = int(ExpParamObj.get('numLevels'))
        self.batch_size =  int(ExpParamObj.get('batch_size'))
        self.tolpart = int(ExpParamObj.get('tolpart'))
        self.m = float(self.batch_size*self.numLevels)
        print 'training samples*numLevels = ',self.m,'number of sample per batch',self.batch_size
        
        



    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data)
        top[0].reshape(1)
 

 
    def forward(self, bottom, top): 
        #weights can be accessed from self.blobs[0].data
        INF = 1e+10          
        labels= bottom[1].data.squeeze()
        print labels
        yi_thisbatch,pyras = self.label_to_pyras(self.batch_size,labels)
        #print bottom[0].data.shape
        #print 'working on ', pyras[0,0]['ith'],'th sample'
        respmaps = bottom[0].data.copy()
        #print self.batch_size
        #print yi_thisbatch
        #print respmaps.shape
        #print 'ith ',pyras[0,0]['ith'][0,0]
        wloss = self.blobs[0].data  
        model = self.vec2model_defbiasonly(wloss,self.model)
        
       

        sum_delta_phi = np.zeros(self.wloss_len,dtype='float64')
        n_lb,n_f,n_y,n_x = respmaps.shape
        slack = np.zeros(self.batch_size,dtype='float64')
        yhatibatch_part_dobj_dRespmaps = np.zeros_like(respmaps)
        yibatch_part_dobj_dRespmaps = np.zeros_like(respmaps) 
        phase = pyras[0,0]['phase'][0,0]           
        for ibatch in range(self.batch_size):
            #print 'phase',pyras[0,ibatch]['phase'][0,0]
            yi = yi_thisbatch[ibatch,0,:,:]
            print '===== NEW ibatch',ibatch,'yi',yi
            loss_y_yhat,maxval,lmaxval,inf_yi_boxes,inf_yhat_boxes,opt_level,lopt_level,ex_i,ex_li=self.learndetect500pyresp(respmaps,pyras,ibatch,model,self.numLevels,-np.inf,yi,0.5,phase)
            bxconvssvm_len,its3 = inf_yi_boxes.shape
            lbxconvssvm_len,its3 = inf_yhat_boxes.shape
            if lbxconvssvm_len == 0 or bxconvssvm_len == 0:
                continue
            infered_yi = np.zeros((self.tolpart,4),dtype='float64')
            for kp in range(self.tolpart):                
                scale = pyras[0,ibatch]['scales'][opt_level,0]
                print scale,pyras[0,ibatch]['padx'][0,0],inf_yi_boxes[kp,1]
                x1  = (inf_yi_boxes[kp,2] - pyras[0,ibatch]['padx'][0,0])*scale+1
                y1  = (inf_yi_boxes[kp,1] - pyras[0,ibatch]['pady'][0,0])*scale+1
                x2  = x1 + 5*scale - 1
                y2  = y1 + 5*scale - 1
                infered_yi[kp,:] = np.array([x1, y1, x2, y2])
            #print 'infered yi',infered_yi,'at level ',opt_level
            print 'inf_yhat_boxes',inf_yhat_boxes
            print 'inf_yi_boxes',inf_yi_boxes
            print 'infered_yi',infered_yi
            print 'labled yi',yi
            x_y_feat_i = np.zeros(self.wloss_len,dtype='float64')
            x_yhat_feat_i = np.zeros(self.wloss_len,dtype='float64')
            for b in range(ex_i['blocks'][0,0].size):
                nb = ex_i['blocks'][0,0][0,b]['x'].size
                #print ex_i['blocks'][0,0][0,b]['x']
                i1 = int(ex_i['blocks'][0,0][0,b]['i'])-1
                i2 = i1 + nb 
                #print 'i1,i2,nb,ex(b).x',i1,i2,nb,ex_i['blocks'][0,0][0,b]['x']
                x_y_feat_i[i1:i2]   = np.reshape(ex_i['blocks'][0,0][0,b]['x'],nb)
            for b in range(ex_li['blocks'][0,0].size):
                nb = ex_li['blocks'][0,0][0,b]['x'].size
                #print ex_i['blocks'][0,0][0,b]['x']
                i1 = int(ex_li['blocks'][0,0][0,b]['i'])-1
                i2 = i1 + nb 
                #print 'i1,i2,nb,ex(b).x',i1,i2,nb,ex_i['blocks'][0,0][0,b]['x']
                x_yhat_feat_i[i1:i2]   = np.reshape(ex_li['blocks'][0,0][0,b]['x'],nb)
            ##print 'size lx',lx.shape

            #print inf_yi_boxes.shape
            #print respmaps.shape

            #bxconvssvm_len,its3 = inf_yi_boxes.shape
            #lbxconvssvm_len,its3 = inf_yhat_boxes.shape
            score_yhat = np.float64(0)
            score_yi = np.float64(0)

            for part_l in range(lbxconvssvm_len):
                convssvm_filter_th = np.sum(self.K[0:part_l])+inf_yhat_boxes[part_l,0]
                #print ibatch*self.numLevels+lopt_level,convssvm_filter_th,inf_yhat_boxes[part_l,1],inf_yhat_boxes[part_l,2]
                yhatibatch_part_dobj_dRespmaps[ibatch*self.numLevels+lopt_level,convssvm_filter_th,inf_yhat_boxes[part_l,1],inf_yhat_boxes[part_l,2]]=1
                score_yhat += respmaps[ibatch*self.numLevels+lopt_level,convssvm_filter_th,inf_yhat_boxes[part_l,1],inf_yhat_boxes[part_l,2]] 
            for part_l in range(bxconvssvm_len):
                convssvm_filter_th = np.sum(self.K[0:part_l])+inf_yi_boxes[part_l,0]
                yibatch_part_dobj_dRespmaps[ibatch*self.numLevels+opt_level,convssvm_filter_th,inf_yi_boxes[part_l,1],inf_yi_boxes[part_l,2]]=1 
                score_yi += respmaps[ibatch*self.numLevels+opt_level,convssvm_filter_th,inf_yi_boxes[part_l,1],inf_yi_boxes[part_l,2]]
            
            #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/learning/SSVM/test_correctness/x_y_feat_python.mat', mdict={'x_y_feat_ihdf5': x_y_feat_i,'x_yhat_feat_ihdf5': x_yhat_feat_i,'inf_yhat_boxes':inf_yhat_boxes,'inf_yi_boxes':inf_yi_boxes,'lopt_level':lopt_level,'opt_level':opt_level,'loss_y_yhat':loss_y_yhat})
            slack[ibatch] = loss_y_yhat  + np.dot( wloss,(x_yhat_feat_i - x_y_feat_i)).squeeze() + score_yhat - score_yi
            print 'score_yhat:',score_yhat
            print 'score_yi:',score_yi
            #slack[ibatch] = score_yhat - score_yi
            #print slack[ibatch],self.svm_C,self.m
            print 'ith' ,pyras[0,ibatch]['ith'][0][0],'slack_loss',slack[ibatch],'loss_y_yhat',loss_y_yhat
            sum_delta_phi = sum_delta_phi + (x_yhat_feat_i - x_y_feat_i)
            #print "sample %d th has loss_yi_yhat  %4.8f" %(pyras[0,ibatch]['ith'][0,0],loss_y_yhat)
        #end for ibatch
        #print yi.shape,infered_yi.shape
        if phase==1:
            Loss = 1.0/self.batch_size*slack.sum()# + np.dot(self.blobs[0].data,self.blobs[0].data)
            
            print "Minibatch lost: %4.10f, ending minibatch at sample %d, its loss_yi_yhat =%4.6f" %(Loss,pyras[0,ibatch]['ith'][0,0],loss_y_yhat)
        else:
            Loss = self.testboxesoverlap(infered_yi,yi) 
            print "Testing differs infered_yi,yi = %4.6f" %(Loss)
        
        

        top[0].data[...] = Loss
        #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/learning/SSVM/test_correctness/LossFeat_python.mat', mdict={'Loss': Loss})
        self.dobj_dRespmaps = 1.0/self.batch_size*(yhatibatch_part_dobj_dRespmaps-yibatch_part_dobj_dRespmaps)
        self.grad_wloss = 1.0/self.batch_size*sum_delta_phi
        #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/learning/SSVM/test_correctness/grad_wlosspython.mat', mdict={'grad_wlosshdf5': self.grad_wloss,'dobj_dRespmapshdf5': self.dobj_dRespmaps,'wsol_py':wloss})
        #print '---------------------------end forward()------------------------------'
        

    def backward(self, top, propagate_down, bottom):
        #if  not propagate_down[0] or propagate_down[1]:
        #    raise Exception("Back prop setting error.")
        
        bottom[0].diff[...] = self.dobj_dRespmaps 
        self.blobs[0].diff[...] = self.grad_wloss
        scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/learning/SSVM/test_correctness/grad_wlosspython_lmdb.mat', mdict={'grad_wlosslmdb_bw': self.blobs[0].diff,'dobj_dRespmapslmdb_bw': bottom[0].diff})
        #print(self.blobs[0].diff)#= np.ones([10,10],dtype=np.float)


    def label_to_pyras(self,batch_size,label_pyra):
    #label_pyra = solver.net.blobs['label'].data
        pyras = np.zeros([1,batch_size],dtype=[('scales', 'O'),('levels','O'), ('level_sizes', 'O'), ('imsize', 'O'), \
('num_levels', 'O'), ('stride', 'O'), ('valid_levels', 'O'), ('padx', 'O'), ('pady', 'O'),('ith', 'O'), ('phase', 'O')])
        yi = np.zeros([batch_size,self.numLevels,self.tolpart,4])
        for bch in range(batch_size):
            bch_numL = bch*self.numLevels    
            print bch,batch_size,self.numLevels,  label_pyra[bch_numL:bch_numL+self.numLevels,4*self.tolpart].shape
            pyras[0,bch]['scales'] = label_pyra[bch_numL:bch_numL+self.numLevels,4*self.tolpart].reshape((self.numLevels,1))
            pyras[0,bch]['levels'] = label_pyra[bch_numL:bch_numL+self.numLevels,4*self.tolpart+1].reshape((self.numLevels,1))
            pyras[0,bch]['imsize'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+2:4*self.tolpart+4].reshape((1,2)).astype(int)
            pyras[0,bch]['level_sizes'] = label_pyra[bch_numL:bch_numL+self.numLevels,4*self.tolpart+4:4*self.tolpart+6].reshape((self.numLevels,2)).astype(int)
            pyras[0,bch]['num_levels'] = np.array([[self.numLevels]],dtype=int)
            pyras[0,bch]['stride'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+6:4*self.tolpart+7].reshape((1,1)).astype(int)
            pyras[0,bch]['valid_levels'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+7:4*self.tolpart+8].reshape((1,1)).astype(int)
            pyras[0,bch]['padx'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+8:4*self.tolpart+9].reshape((1,1)).astype(int)
            pyras[0,bch]['pady'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+9:4*self.tolpart+10].reshape((1,1)).astype(int)
            pyras[0,bch]['ith'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+10:4*self.tolpart+11].reshape((1,1)).astype(int)
            pyras[0,bch]['phase'] = label_pyra[bch_numL:bch_numL+1,4*self.tolpart+11:4*self.tolpart+12].reshape((1,1)).astype(int)
            yi[bch,:,:,:] = label_pyra[bch_numL:bch_numL+self.numLevels,0:4*self.tolpart].reshape((self.numLevels,self.tolpart,4))
        
        #print pyras[0,bch]['padx'],pyras[0,bch]['pady'],pyras[0,0]['ith'] 
        #print yi[0,3,:,:]
        return yi,pyras
    
    def vec2model_defbiasonly(self,w,model):

        for i in range(model['bias'][0,0].size):
            s = model['bias'][0,0][0,i]['w'].shape
            j = np.arange(model['bias'][0,0][0,i]['i'][0,0],model['bias'][0,0][0,i]['i'][0,0]+model['bias'][0,0][0,i]['w'].size)-1
            #print j
            model['bias'][0,0][0,i]['w'] = np.reshape(w[j],s)
        
        for i in range(model['defs'][0,0].size):
            s = model['defs'][0,0][0,i]['w'].shape
            j = np.arange(model['defs'][0,0][0,i]['i'][0,0],model['defs'][0,0][0,i]['i'][0,0]+model['defs'][0,0][0,i]['w'].size)-1
            #print j , model['defs'][0,0][0,i]['w'].shape
            model['defs'][0,0][0,i]['w'] = np.reshape(w[j],s)
        
        return model

    def model2vec_defbiasonly(self,model):
        '''w = model2vec_defbiasonly(model)'''    
        w  = np.zeros([int(model[0,0]['len'])],dtype='float64')
        for i in range(model['bias'][0,0].size):
            j = np.arange(model['bias'][0,0][0,i]['i'][0,0],model['bias'][0,0][0,i]['i'][0,0]+model['bias'][0,0][0,i]['w'].size)-1
            #print j
            w[j] = model['bias'][0,0][0,i]['w']

        for i in range(model['defs'][0,0].size):
            j = np.arange(model['defs'][0,0][0,i]['i'][0,0],model['defs'][0,0][0,i]['i'][0,0]+model['defs'][0,0][0,i]['w'].size)-1
            #print  model['defs'][0,0][0,i]['w'].shape
            #print w[j].shape
            w[j] = model['defs'][0,0][0,i]['w'].squeeze()
        #print w.shape
        return w

    def testoverlap(self,sizx,sizy,pyra,level,bbox,overlap,tolpart):
        '''function [ovmask ovvalue] = testoverlap(sizx,sizy,pyra,level,bbox,overlap,tolpart)'''
        scale = float(pyra['scales'][level,0])
        padx  = int(pyra['padx'][0,0])
        pady  = int(pyra['pady'][0,0])
        dimy =  int(pyra['level_sizes'][level,0])
        dimx =  int(pyra['level_sizes'][level,1])

        #print dimy,dimx,padx,pady,scale
        #print bbox[k][0][0,0]
        bx1 = bbox[0]
        by1 = bbox[1]
        bx2 = bbox[2]
        by2 = bbox[3]
        #print bbox[0,0]
        #if level==3:
        #    print bx1,by1,bx2,by2


        # % Index windows evaluated by filter (in image coordinates)
        x1 = (np.arange(1,dimx-sizx+2) - padx - 1)*scale + 1;
        y1 = (np.arange(1,dimy-sizy+2) - pady - 1)*scale + 1;
        x2 = x1 + sizx*scale - 1;
        y2 = y1 + sizy*scale - 1;
        #print x1.shape,y1.shape,x2.shape,y2.shape

        # % Compute intersection with bbox
        xx1 =  np.maximum(x1,bx1)
        xx2 = np.minimum(x2,bx2)
        yy1 = np.maximum(y1,by1)
        yy2 = np.minimum(y2,by2)
        w   = xx2 - xx1 + 1
        h   = yy2 - yy1 + 1
        w[w<0] = 0
        h[h<0] = 0
        h2dtranspose = np.expand_dims(h, axis=1)
        w2d = np.expand_dims(w, axis=0)
        inter  = h2dtranspose*w

        #% area of (possibly clipped) detection windows and original bbox
        area = np.expand_dims((y2-y1+1),axis=1)*np.expand_dims((x2-x1+1),axis=0)
        box  = float(np.expand_dims((by2-by1+1),axis=1)*np.expand_dims((bx2-bx1+1),axis=0))

        ovmask   = (np.divide(inter , (area + box - inter))>overlap).astype('int8')
        ovvalue = (1- np.divide(inter,(area + box - inter)))/tolpart
        #ovmask = np.zeros_like(ovmask,dtype='bool')
        #ovmask[np.flatnonzero(ovmask_)] = True
        return ovmask, ovvalue

    def testboxesoverlap(self,bbox1,bbox2):
        numparts,four_surely = bbox1.shape
        b1x1 = bbox1[:,0]
        b1y1 = bbox1[:,1]
        b1x2 = bbox1[:,2]
        b1y2 = bbox1[:,3]
  
        b2x1 = bbox2[:,0]
        b2y1 = bbox2[:,1]
        b2x2 = bbox2[:,2]
        b2y2 = bbox2[:,3]
        # % Compute intersection with bbox
        xx1 = np.maximum(b1x1,b2x1)
        xx2 = np.minimum(b1x2,b2x2)
        yy1 = np.maximum(b1y1,b2y1)
        yy2 = np.minimum(b1y2,b2y2)
        w   = xx2 - xx1 + 1;
        h   = yy2 - yy1 + 1;
        w[w<0] = 0;
        h[h<0] = 0;
        h2dtranspose = np.expand_dims(h, axis=1)
        w2d = np.expand_dims(w, axis=0)
        inter  = h2dtranspose*w;

        #area of (possibly clipped) detection windows and original bbox
        area_b1 =np.multiply((b1y2-b1y1+1),(b1x2-b1x1+1))
        area_b2 =np.multiply((b2y2-b2y1+1),(b2x2-b2x1+1))
  
        # thresholded overlap
        one_minus_inter_dvded_by_union_boxes  =np.divide( 1- inter , (area_b1 + area_b2 - inter ) )/numparts
        one_minus_inter_dvded_by_union = np.sum( one_minus_inter_dvded_by_union_boxes)

        return one_minus_inter_dvded_by_union



    def modelcomponents_nofilter(self,model,pyras,ibatch,K):
        components = np.zeros([1,1],dtype='O')
        numcomponents = model['components'][0,0][0,0].size
        components[0,0] = np.zeros([1,numcomponents],dtype=[('biasid', 'O'), ('defid', 'O'),  \
                                                        ('parent', 'O'), ('sizy', 'O'), ('sizx', 'O'), ('w', 'O'), \
                                                        ('defI', 'O'), ('starty', 'O'), ('startx', 'O'), ('step', 'O'),\
                                                        ('level', 'O'), ('Ix', 'O'), ('Iy', 'O'), ('Ik', 'O'), ('scale', 'O'),\
                                                        ('b', 'O'), ('biasI', 'O'), ('score','O')])
    
     
        for k in range(numcomponents):      
            #print 'k=k',k
            components[0,0][0,k]['biasid']= model['components'][0,0][0,0][0]['biasid'][k]
            components[0,0][0,k]['defid'] = model['components'][0,0][0,0][0]['defid'][k]
            #components[0,0][0,k]['filterid'] = model['components'][0,0][0,0][0]['filterid'][k]
            components[0,0][0,k]['parent'] = model['components'][0,0][0,0][0]['parent'][k]

            #      % store the scale of each part relative to the component root
            par = components[0,0][0,k]['parent'][0,0]

       
            biasnumrow,biasnumcol=components[0,0][0,k]['biasid'].shape
            onearray_b = np.zeros(biasnumrow*biasnumcol,dtype='float64')
            onearray_biasI = np.zeros(biasnumrow*biasnumcol,dtype='float64')
            components[0,0][0,k]['b'] = model['bias'][0,0][0,components[0,0][0,k]['biasid'].reshape(components[0,0][0,k]['biasid'].size,1)-1]['w']
            for i in range(components[0,0][0,k]['b'].size): 
                #print components[0,0][0,k]['b'][i].shape
                onearray_b[i] = np.asscalar(components[0,0][0,k]['b'][i]) 
            components[0,0][0,k]['b'] = np.reshape(onearray_b,(1,biasnumrow,biasnumcol))
       
            components[0,0][0,k]['biasI'] = model['bias'][0,0][0,components[0,0][0,k]['biasid'].reshape(components[0,0][0,k]['biasid'].size,1)-1]['i']
            for i in range(components[0,0][0,k]['biasI'].size): 
                #print components[0,0][0,k]['biasI'][i].shape
                onearray_biasI[i] = np.asscalar(components[0,0][0,k]['biasI'][i])        
            components[0,0][0,k]['biasI'] = np.reshape(onearray_biasI,(biasnumrow,biasnumcol))    
            components[0,0][0,k]['sizy'] = K[k]+np.zeros([1,K[k]],dtype='uint8')#np.zeros([1,components[0,0][0,k]['filterid'].size],dtype='uint8')
            components[0,0][0,k]['sizx'] = K[k]+np.zeros([1,K[k]],dtype='uint8')#np.zeros([1,components[0,0][0,k]['filterid'].size],dtype='uint8')
        #components[0,0][0,k]['filterI'] = np.zeros([1,components[0,0][0,k]['filterid'].size],dtype='float64')
        #for f in range(components[0,0][0,k]['filterid'].size):
        #    print components[0,0][0,k]['filterid'][0,f]
        #    x = model['filters'][0,0][0,components[0,0][0,k]['filterid'][0,f]-1]
        #    print x['i']
        #    components[0,0][0,k]['filterI'][0,f] = x['i'][0,0]
        #    print components[0,0][0,k]['filterI']
        #    components[0,0][0,k]['sizy'][0,f], components[0,0][0,k]['sizx'][0,f],foo = x['w'].shape            

            components[0,0][0,k]['w'] = np.zeros([4,components[0,0][0,k]['defid'].size],dtype='float64')
            components[0,0][0,k]['defI'] = np.zeros([components[0,0][0,k]['defid'].size],dtype='uint32')
            components[0,0][0,k]['starty'] = np.zeros([components[0,0][0,k]['defid'].size],dtype='float64')
            components[0,0][0,k]['startx'] = np.zeros([components[0,0][0,k]['defid'].size],dtype='float64')
            for f in range(components[0,0][0,k]['defid'].size):        
                x = model['defs'][0,0][0,components[0,0][0,k]['defid'][0,f]-1]           
                components[0,0][0,k]['w'][:,f]  = x['w'].T.squeeze() 
                components[0,0][0,k]['defI'][f] = x['i']
                ax  = x['anchor'][0,0]
                ay  = x['anchor'][0,1]   
                ds  = x['anchor'][0,2]
                components[0,0][0,k]['scale'] = ds 
                # amount of (virtual) padding to hallucinate
                step     = 2**ds
                virtpady = (step-1)*pyras['pady'][0,ibatch]
                virtpadx = (step-1)*pyras['padx'][0,ibatch]
                #% starting points (simulates additional padding at finer scales)
                components[0,0][0,k]['starty'][f] = ay-virtpady;
                components[0,0][0,k]['startx'][f] = ax-virtpadx;      
                components[0,0][0,k]['step']   = step;
         
        return components



    def modelcomponents(self,model,pyras,ibatch):
        components = np.zeros([1,1],dtype='O')
        numcomponents = model['components'][0,0][0,0].size
        components[0,0] = np.zeros([1,numcomponents],dtype=[('biasid', 'O'), ('defid', 'O'), ('filterid', 'O'), \
                                                        ('parent', 'O'), ('sizy', 'O'), ('sizx', 'O'), ('w', 'O'), \
                                                        ('defI', 'O'), ('starty', 'O'), ('startx', 'O'), ('step', 'O'),\
                                                        ('level', 'O'), ('Ix', 'O'), ('Iy', 'O'), ('Ik', 'O'), ('scale', 'O'),\
                                                        ('b', 'O'), ('biasI', 'O'), ('filterI', 'O'),('score','O')])
    
     
        for k in range(numcomponents):      
       
            components[0,0][0,k]['biasid']= model['components'][0,0][0,0][0]['biasid'][k]
            components[0,0][0,k]['defid'] = model['components'][0,0][0,0][0]['defid'][k]
            components[0,0][0,k]['filterid'] = model['components'][0,0][0,0][0]['filterid'][k]
            components[0,0][0,k]['parent'] = model['components'][0,0][0,0][0]['parent'][k]

            #      % store the scale of each part relative to the component root
            par = components[0,0][0,k]['parent'][0,0]
            #print 'assert par<k: par=',par,'k=',k
            assert par <= k
       
            biasnumrow,biasnumcol=components[0,0][0,k]['biasid'].shape
            onearray_b = np.zeros(biasnumrow*biasnumcol)
            onearray_biasI = np.zeros(biasnumrow*biasnumcol)
            components[0,0][0,k]['b'] = model['bias'][0,0][0,components[0,0][0,k]['biasid'].reshape(components[0,0][0,k]['biasid'].size,1)-1]['w']
            for i in range(components[0,0][0,k]['b'].size): 
                #print components[0,0][0,k]['b'][i].shape
                onearray_b[i] = np.asscalar(components[0,0][0,k]['b'][i]) 
            components[0,0][0,k]['b'] = np.reshape(onearray_b,(1,biasnumrow,biasnumcol))
       
            components[0,0][0,k]['biasI'] = model['bias'][0,0][0,components[0,0][0,k]['biasid'].reshape(components[0,0][0,k]['biasid'].size,1)-1]['i']
            for i in range(components[0,0][0,k]['biasI'].size): 
                #print components[0,0][0,k]['biasI'][i].shape
                onearray_biasI[i] = np.asscalar(components[0,0][0,k]['biasI'][i])        
            components[0,0][0,k]['biasI'] = np.reshape(onearray_biasI,(biasnumrow,biasnumcol))    
            components[0,0][0,k]['sizy'] = np.zeros([1,components[0,0][0,k]['filterid'].size],dtype='uint8')
            components[0,0][0,k]['sizx'] = np.zeros([1,components[0,0][0,k]['filterid'].size],dtype='uint8')
            components[0,0][0,k]['filterI'] = np.zeros([1,components[0,0][0,k]['filterid'].size],dtype='float64')
            for f in range(components[0,0][0,k]['filterid'].size):
                #print components[0,0][0,k]['filterid'][0,f]
                x = model['filters'][0,0][0,components[0,0][0,k]['filterid'][0,f]-1]
                #print x['i']
                components[0,0][0,k]['filterI'][0,f] = x['i'][0,0]
                #print components[0,0][0,k]['filterI']
                components[0,0][0,k]['sizy'][0,f], components[0,0][0,k]['sizx'][0,f],foo = x['w'].shape            

            components[0,0][0,k]['w'] = np.zeros([4,components[0,0][0,k]['defid'].size],dtype='float64')
            components[0,0][0,k]['defI'] = np.zeros([components[0,0][0,k]['defid'].size],dtype='uint32')
            components[0,0][0,k]['starty'] = np.zeros([components[0,0][0,k]['defid'].size],dtype='float64')
            components[0,0][0,k]['startx'] = np.zeros([components[0,0][0,k]['defid'].size],dtype='float64')
            for f in range(components[0,0][0,k]['defid'].size):        
                x = model['defs'][0,0][0,components[0,0][0,k]['defid'][0,f]-1]           
                components[0,0][0,k]['w'][:,f]  = x['w'].T.squeeze() 
                components[0,0][0,k]['defI'][f] = x['i']
                ax  = x['anchor'][0,0]
                ay  = x['anchor'][0,1]   
                ds  = x['anchor'][0,2]
                components[0,0][0,k]['scale'] = ds 
                # amount of (virtual) padding to hallucinate
                step     = 2**ds
                virtpady = (step-1)*pyras['pady'][0,ibatch]
                virtpadx = (step-1)*pyras['padx'][0,ibatch]
                #% starting points (simulates additional padding at finer scales)                
                components[0,0][0,k]['starty'][f] = ay-virtpady;
                components[0,0][0,k]['startx'][f] = ax-virtpadx;      
                components[0,0][0,k]['step']   = step;

         
        return components

    def dt2d(self,image2d,ax,bx,ay,by):
        '''[score_ans,Ix_ans,Iy_ans] = dt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k));'''
        nrow,ncol = image2d.shape
        #for x in range(ncol):
        score_x,Ixy = dt.compute(image2d,axes=(0,1),f=dt.L2(-ax,-bx,-ay,-by))
        Ix = Ixy[1]
        Iy = Ixy[0]
        #for y in range(nrow):
            #score,Iy = dt.compute(score_x, axes=(0,0),f=dt.L2(-ay,-by))
        
        return score_x,Ix,Iy


    def passmsg(self,child_part,parent_part):
        '''function [score,Ix,Iy,Ik] = passmsg(child,parent)'''
    
        INF = 1e+10
        K ,  = child_part['filterid'].squeeze().shape 
        Nk,Ny,Nx = parent_part['score'].shape
        Ix0 = np.zeros([K,Ny,Nx])
        Iy0 = np.zeros([K,Ny,Nx])
        score0 = np.zeros([K,Ny,Nx],dtype='float64') - INF
        L = parent_part['filterid'].size
        for k in range(K):
            #assert child_part['w'][0,k]>0
            #assert child_part['w'][2,k]>0
            #print child_part['w'][0,k],child_part['w'][1,k],child_part['w'][2,k]
            #score_tmp,Ix_tmp,Iy_tmp = self.dt2d(child_part['score'][k,:,:].astype('float64'),child_part['w'][0,k],child_part['w'][1,k],child_part['w'][2,k],child_part['w'][3,k])
            #print child_part['score'][k,:,:].flags

            score_tmp,Ix_tmp,Iy_tmp=mahotas.distance(np.asfortranarray(child_part['score'][k,:,:]),child_part['w'][0,k],child_part['w'][1,k],child_part['w'][2,k],child_part['w'][3,k])
            startx = np.int_(child_part['startx'][k])-1
            starty = np.int_(child_part['starty'][k])-1
            step = child_part['step'].squeeze()
            #% ending points
            endy = starty+step*(Ny-1)+1
            endx = startx+step*(Nx-1)+1    
            endy = min(child_part['score'].shape[1],endy)
            endx = min(child_part['score'].shape[2],endx)
            #y sample points
            iy = np.arange(starty,endy,step).astype(int)
            ix = np.arange(startx,endx,step).astype(int)
            #print 'size iy,ix', iy.size,ix.size
            if iy.size == 0 or ix.size ==0:            
                score = np.zeros([L,Ny,Nx],dtype='float64')
                Ix = np.zeros([L,Ny,Nx]).astype(int)
                Iy = np.zeros([L,Ny,Nx]).astype(int)
                Ik = np.zeros([L,Ny,Nx]).astype(int)
                return score,Ix,Iy,Ik
            oy = sum(iy<0)
            #print 'iy',iy
            iy = iy[iy>=0]              
            ox = sum(ix<0)
            ix = ix[ix>=0]          
            #sample scores        
            sp = score_tmp[iy,:][:,ix]        
            sx = Ix_tmp[iy,:][:,ix]
            sy = Iy_tmp[iy,:][:,ix]
            sz = sp.shape    
            #define msgs
            iy  = np.arange(oy,oy+sz[0])
            ix  = np.arange(ox,ox+sz[1]) 
            iyp = np.tile(iy,[ix.shape[0],1]).T
            ixp = np.tile(ix,[iy.shape[0],1])
            score0[k,iyp,ixp] = sp
            Ix0[k,iyp,ixp] = sx;
            Iy0[k,iyp,ixp] = sy;  
        
        
        #% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K    
        N  = Nx*Ny;
        i0 = np.arange(N).reshape(Ny,Nx) 
        i0row,i0col = np.unravel_index(i0, [Ny,Nx])
        score = np.zeros([L,Ny,Nx],dtype='float64')
        Ix = np.zeros([L,Ny,Nx]).astype(int)
        Iy = np.zeros([L,Ny,Nx]).astype(int)
        Ik = np.zeros([L,Ny,Nx]).astype(int)
        for l in range(L):
            b = child_part['b'][0,l,:]
            score0b = (score0.transpose([1,2,0])+b).transpose([2,0,1])
            score[l,:,:]= score0b.max(axis=0)
            I = score0b.argmax(axis=0)    
            #print  Ix.shape, Ix0.shape,  i0row.shape,i0col.shape,I.shape  
            Ix[l,:,:]  = Ix0[I,i0row,i0col]
            Iy[l,:,:]  = Iy0[I,i0row,i0col]
            Ik[l,:,:]    = I

        return score,Ix,Iy,Ik


    def defvector(self,px,py,x,y,mix,part):
        '''Compute the deformation feature given parent locations, 
        child locations, and the child part
        function res = defvector(px,py,x,y,mix,part)'''
        #print part['step']
        #print  part['startx'][mix] 
        probex = (px-1)*part['step'] + part['startx'][mix] 
        probey = (py-1)*part['step'] + part['starty'][mix] 
        dx = probex - x
        dy = probey - y
        res = -1*np.array([[dx*dx, dx, dy*dy, dy]],dtype='float64').T

        return res


    def backtrack(self,x,y,mix,parts,level):#,padx,pady,scale):
        '''function [box,ex] = testingbacktrack(x,y,mix,parts,pyra,ex,write)'''


        tolpart = parts.size
        #print 'tolpart',tolpart
        ptr = np.zeros([tolpart,3]) 
        k   = 0
        p   = parts[0,k]
        ptr[k,:] = np.array([mix, y, x]) 
        sizx=int(parts[0,k]['sizx'][0,0])
        sizy=int(parts[0,k]['sizy'][0,0])
        ex = np.zeros([1,1],dtype=[('blocks', 'O'), ('id', 'O')])
        ex['blocks'][0,0] = np.zeros([1,1],dtype=[('i', 'O'), ('x', 'O'), ('type', 'O'), ('level', 'O'), ('box', 'O')])
        ex['id'][0,0]=np.zeros([1,5],dtype='uint8')
    
        ex['id'][0,0][0,2:5]= np.array([level, round(x+sizx/2), round(y+sizy/2)]);
        ex['blocks'][0,0]=np.zeros([1,1],dtype=[('i', 'O'), ('x', 'O'),('type', 'O'),('level', 'O'), ('box', 'O')])
        ex['blocks'][0,0]['i'][0,0] =  p['biasI']
        ex['blocks'][0,0]['x'][0,0] = np.ones([1,1],dtype='float64')
        ex['blocks'][0,0]['type'][0,0]=np.array(['bias'],dtype='<U4')
        ex['blocks'][0,0]['level'][0,0] = np.array([],dtype='float64')
        #ex['blocks'][0,0]['box'][0,0] = np.array([],dtype='float64')
        #ex['blocks'][0,0].resize((1,2))
        #ex['blocks'][0,0]['i'][0,1]=np.zeros([1,1],dtype=int) + p['filterI'][0,mix]
        #ex['blocks'][0,0]['x'][0,1]=pyra['feat'][0,0][level][0][:,y:y+sizy,x:x+sizx].transpose([1,2,0])
        #ex['blocks'][0,0]['type'][0,1]=np.array(['bias'],dtype='<U4')
        #ex['blocks'][0,0]['level'][0,1] = np.array([1,1],dtype='float64')+level
        #ex['blocks'][0,0]['box'][0,1] = np.array([[x,y,x+sizx-1,y+sizy-1]],dtype='float64')    
    
    
        for k in range(1,tolpart):
            p   = parts[0,k]              
            #sizx = int(p['sizx'][0,mix])
            #sizy = int(p['sizy'][0,mix])
            par = p['parent']-1
            #print par
            mix = ptr[par,0][0,0]
            y = ptr[par,1][0,0]
            x = ptr[par,2][0,0]
        
        
            #print 'parts',k,'mix',mix,'y',y,'x',x
            ptr[k,0] = p['Ik'][mix,y,x]
            ptr[k,1] = p['Iy'][mix,y,x]         
            ptr[k,2] = p['Ix'][mix,y,x]  

            bsize = ex['blocks'][0,0].size
            ex['blocks'][0,0].resize((1,bsize+2))     
            ex['blocks'][0,0]['i'][0,bsize]= p['biasI'][mix,ptr[k,0]]
            ex['blocks'][0,0]['x'][0,bsize]= np.ones([1,1],dtype='float64')
            ex['blocks'][0,0]['type'][0,bsize]=np.array(['bias'],dtype='<U4')
            #print  p['defI'].shape
            ex['blocks'][0,0]['i'][0,bsize+1]= p['defI'][ptr[k,0]]
            ex['blocks'][0,0]['x'][0,bsize+1]= self.defvector(x,y,ptr[k,2],ptr[k,1],ptr[k,0],p)
            ex['blocks'][0,0]['type'][0,bsize+1]=np.array(['defs'],dtype='<U4')
     
        return ptr,ex


    def backtracklossscore(self,x,y,mix,parts,lossscoreparts):
        '''function [box,ex] = testingbacktrack(x,y,mix,parts,lossscoreparts,pyra,ex)'''
        ptr = np.zeros([self.tolpart,3])
        lossbx = np.zeros(self.tolpart)
        for k in range(1,self.tolpart):
            p   = parts[0,k]
            par = p['parent']-1
            mix = ptr[par,0][0,0]
            y = ptr[par,1][0,0]
            x = ptr[par,2][0,0]
       
            ptr[k,0] = p['Ik'][mix,y,x]
            ptr[k,1] = p['Iy'][mix,y,x]
            ptr[k,2] = p['Ix'][mix,y,x] 
            foundloss = lossscoreparts['score'][0,k][ptr[k,0],ptr[k,1],ptr[k,2]]
            lossbx[k] = foundloss
            #print scale
            #x1 = (ptr[k,0]-int(pyra['padx']))*scale+1
            #y1 = (ptr[k,1]-int(pyra['padx']))*scale+1
            #x2  = x1 + p['sizx'][0,ptr[k,2]]*scale -1      
            #y2  = y1 + p['sizy'][0,ptr[k,2]]*scale -1  
            #box[k,:] = [x1,y1,x2,y2]       
      
     
        return lossbx


    def learndetect500pyresp(self,respmaps,pyras,ibatch,model,numLevels,thresh,yi,overlap,phase):
        '''ibatch is ith sample in this batch accessed by pyras[0,ibatch], the range of which is within batch_size'''
        firsttime_save = True;
        INF = 1e+10
        maxval= -INF
        lmaxval = -INF
        lthresh = -INF
        thresh = -INF
        loss_y_yhat =-INF
        inf_yhat_boxes = np.empty( shape=(0, 0) )
        inf_yi_boxes = np.empty( shape=(0, 0) )
        ex_i = np.zeros([1,1],dtype=[('blocks', 'O'), ('id', 'O')])
        ex_i['blocks'][0,0] = np.empty( shape=(0, 0) )
        ex_li = np.zeros([1,1],dtype=[('blocks', 'O'), ('id', 'O')])
        ex_li['blocks'][0,0] = np.empty( shape=(0, 0) )
        opt_level=-1
        lopt_level =-1 
        WRITE = 0
        #saved_flag = False;
        components =  self.modelcomponents(model,pyras,ibatch)    
        #y_im_i = yi[0,0,:,:].squeeze()
        lcnt = 0
        #lossbx_each = 1/3*np.ones((3,),dtype='float64')
        #print 'pyra levelsizes',pyras[0,ibatch]['level_sizes']
        for rlevel in range(numLevels):
            numcomponents = model['components'][0,0].size
            #level_sizes in [[level,nrow,ncol]]
            thislevel_level_sizes = pyras[0,ibatch]['level_sizes'][rlevel,:]
            #print 'thislevel_level_sizes',thislevel_level_sizes
            #print 'numcomponents=',numcomponents
            for c in range(numcomponents):
                parts = components[0,c].copy()
                lparts = components[0,c].copy()
                lossscoreparts = components[0,c].copy()
                
                for k in range(self.tolpart):
                    numFilters_before_kthparts = np.sum(self.K[0:k])
                    numFilters_kthparts = self.K[k]
                    sizx=int(parts[0,k]['sizx'][0,0])
                    sizy=int(parts[0,k]['sizy'][0,0])
                    #print sizx, sizy
                    #print 'respmaps[rlevel].shape',respmaps[rlevel+ibatch*self.numLevels,:,:,:].shape
                    thislevel_respmaps = respmaps[rlevel+ibatch*self.numLevels,:,0:thislevel_level_sizes[0]-sizy+1,0:thislevel_level_sizes[1]-sizx+1].copy()
                    #if rlevel==2 and k ==25:
                        #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/Thislevel_respmaps.mat', mdict={'thislevel_respmaps': thislevel_respmaps})
                    #print 'thislevel_respmaps.shape',thislevel_respmaps.shape
                    #thislevel_respmaps = np.pad(thislevel_respmaps, ((0,0),(0,thislevel_level_sizes[0]-thislevel_respmaps.shape[1]-5+1),(0,0)),mode='constant')
                    #thislevel_respmaps = np.pad(thislevel_respmaps, ((0,0),(0,0),(0,thislevel_level_sizes[1]-thislevel_respmaps.shape[2]-5+1)),mode='constant')
                    #print 'thislevel_respmaps.shape',thislevel_respmaps.shape
                    # thislevel_level_sizes[1]-thislevel_respmaps.shape[2])), constant_values=(0, 0))
                    #  FIXME I may pad    pyra.feat{i} = padarray(pyra.feat{i}, [pady padx 0], 0);
                    
                    thispartscore = thislevel_respmaps[numFilters_before_kthparts:numFilters_before_kthparts+numFilters_kthparts,:,:].copy().astype('float64')
                    parts[0,k]['level'] = rlevel
                    parts[0,k]['score'] = np.asfortranarray(thispartscore.copy())
                    if phase==1:
                        lparts[0,k]['level'] = rlevel
                        lossscoreparts[0,k]['level'] = rlevel                    
                        lparts[0,k]['score'] = thispartscore.copy()
                        lossscoreparts[0,k]['score'] = np.zeros_like(thispartscore)
                    #print parts[0,0]['score'][0,0:10,0:10] 
                    #print '================='
                    #print 'numFilters_kthparts=',numFilters_kthparts
                    for fi in range(numFilters_kthparts):
                        if phase == 1:
                            ovmask, lossscore = self.testoverlap(parts[0,k]['sizx'][0,fi],parts[0,k]['sizy'][0,fi],pyras[0,ibatch],rlevel,yi[k,:],overlap,self.tolpart)
                            #if rlevel==2 and k ==25 and fi==1:
                                #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/ovmask_level3_k26_fi1.mat', mdict={'ovmask_py': ovmask})
                            #print 'rlevel:',rlevel,'mask shape',ovmask.shape,'parts shape', parts[0,k]['score'][fi,:,:].shape,'sizx',parts[0,k]['sizx'][0,fi],'sizey',parts[0,k]['sizy'][0,fi],yi[k,:],'nonzero ovmask',np.count_nonzero(ovmask)
                            #if np.count_nonzero(ovmask)>0 and not saved_flag:
                                #print np.nonzero(ovmask)
                                #print parts[0,k]['score'][fi,:,:][ovmask!=0] 
                                #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/partlevel0_fi0_k0.mat', mdict={'partlevel0_fi0_k0': parts,'pyovmask':ovmask,'pyk':k,'pyfi':fi,'thislevel_respmaps':thislevel_respmaps,'numFilters_before_kthparts':numFilters_before_kthparts,'numFilters_before_kthparts':numFilters_before_kthparts,'pyrlevel':rlevel})
                                #saved_flag = True;
                            parts[0,k]['score'][fi,:,:][ovmask==0] = -INF                         
                            lparts[0,k]['score'][fi,:,:] = (lparts[0,k]['score'][fi,:,:] + lossscore).copy()
                            lossscoreparts[0,k]['score'][fi,:,:] = lossscore.copy()
                   

                    #end fi
                ##endfor k 
                #if rlevel==2:
                    #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/partsscore_pyb4.mat', mdict={'partsb4_py':parts})              
                for k in range(self.tolpart-1,0,-1):
                    par = parts[0,k]['parent'].squeeze()-1
                    #print 'par',par,'k',k 
                    #if rlevel == 0 and firsttime_save:
                        #parts_pass = parts.copy()
                    msg,Ix,Iy,Ik = self.passmsg(parts[0,k],parts[0,par])
                    #if rlevel==2 and k==18:
                        #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/msg19_py.mat', mdict={'msg19_py':msg})
                    #print 'Ix,y,k shape',Ix.shape, Iy.shape, Ik.shape
                    parts[0,k]['Ix'] = Ix.copy()
                    parts[0,k]['Iy'] = Iy.copy()
                    parts[0,k]['Ik'] = Ik.copy()
                    parts[0,par]['score'] = (parts[0,par]['score'] + msg).copy()
                    if phase==1:
                        lmsg,lIx,lIy,lIk = self.passmsg(lparts[0,k],lparts[0,par])
                        lparts[0,k]['Ix'] = lIx.copy()
                        lparts[0,k]['Iy'] = lIy.copy()
                        lparts[0,k]['Ik'] = lIk.copy()
                        lparts[0,par]['score'] = lparts[0,par]['score'] + lmsg
                ##endfor k
                #% Add bias to root score
                #if rlevel==2:
                    #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/partsscore_py.mat', mdict={'parts_py':parts})
                parts[0,0]['score'] = (parts[0,0]['score'] + parts[0,0]['b']).copy()
                rscore = parts[0,0]['score'].max(axis=0)
                Ik = parts[0,0]['score'].argmax(axis=0)
                try:
                    thresh = max(thresh,rscore.max())
                    print 'rscore_max ', thresh
                except: 
                    continue
                Y,X = (rscore >= thresh).nonzero()
                for i in range(X.size):
                    x = int(X[i])
                    y = int(Y[i])
                    k = int(Ik[y,x])
                    print 'root at x=',x,',y=',y,',k=',k,',level=',rlevel,'thresh update to',thresh
                    #scipy.io.savemat('MySubGradientSSVM_1_3_is_1_2/pyrscore.mat', mdict={'pyrscore': rscore})
                    opt_level = rlevel
                    inf_yi_boxes,ex_i = self.backtrack(x,y,k,parts,opt_level)
                    maxval = rscore[y,x]
                if phase==1:
                    lparts[0,0]['score'] = (lparts[0,0]['score'] + lparts[0,0]['b']).copy()
                    lrscore = lparts[0,0]['score'].max(axis=0)
                    lIk = lparts[0,0]['score'].argmax(axis=0)                
                    lthresh = max(lthresh,lrscore.max())        
                    lY,lX = (lrscore >= lthresh).nonzero()           
                    for i in range(lX.size):
                        lx = lX[i]
                        ly = lY[i]
                        lk = lIk[ly,lx]
                        print 'lroot at lx=',lx,',ly=',ly,',lk=',lk,',level=',rlevel,'lthresh update to',lthresh
                        lopt_level = rlevel
                        inf_yhat_boxes,ex_li = self.backtrack(lx,ly,lk,lparts,lopt_level)
                        #lossbx_each = self.backtracklossscore(lx,ly,lk,lparts,lossscoreparts)
                        #lossbx_each[0] =  lossscoreparts[0,0]['score'][lk,ly,lx]
                        lossbx_each = np.zeros((self.tolpart,),dtype='float64')
                        for each_loss_i in range(self.tolpart):
                            lossbx_each[each_loss_i] = lossscoreparts[0,each_loss_i]['score'][inf_yhat_boxes[each_loss_i,0],inf_yhat_boxes[each_loss_i,1],inf_yhat_boxes[each_loss_i,2]]

                        print 'lossbx_each',lossbx_each
                        #print 'lossbx_each2',lossbx_each2
                        lcnt = lcnt + 1
                        lmaxval = lrscore[ly,lx]
        
        
                #endfor i                
            #endfor c
        #endfor rlevel
        if phase==1:
            loss_y_yhat = np.sum(lossbx_each)            
        else:
            #loss_y_yhat = testboxesoverlap(infered_yi,yi) 
            inf_yhat_boxes = inf_yi_boxes
            ex_li = ex_i
            lopt_level = opt_level
            lmaxval = maxval
    
        return loss_y_yhat,maxval,lmaxval,inf_yi_boxes,inf_yhat_boxes,opt_level,lopt_level,ex_i,ex_li                


