%inrun_experimentPoseEstParse(0,200,0,205,26,5e-2,1,1e-1,0.5,false);
clear; close all;%!rm PARSE/binlevel/*
addpath(genpath('.'));

numPos = 200;
numNeg = 0;
numTest = 0;
tolpart = 26;
SVM_tol = 5e-3;
SVM_C = 0.01;
SSVM_epsilon=1e-2;
HOG_coordinate_overlap = 0.5;
reload = false;
featdan = 150;
batch_size =5;
sbin=4;
num_test_batches = 0;
num_train_batches = 0;

% --------------------
% specify model parameters
% number of mixtures for 26 parts
K = [5 5 5 6 6 6 6 5 5 5 5 5 5 5 ...
         5 6 6 6 6 5 5 5 5 5 5 5]; 
% Tree structure for 26 parts: pa(i) is the parent of part i
% This structure is implicity assumed during data preparation
% (PARSE_data.m) and evaluation (PARSE_eval_pcp)
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
% Spatial resolution of HOG cell, interms of pixel width and hieght
% The PARSE dataset contains low-res people, so we use low-res parts

K=K(1:tolpart);
pa = pa(1:tolpart);
% --------------------
% Prepare training and testing images and part bounding boxes
% You will need to write custom IMAGE_data() functions for your dataset
globals;
name = 'PARSE';
[pos test] = PARSE_data();
test = test(1:numTest);
neg        = INRIA_data();
pos        = pointtobox(pos,pa);
test = pointtobox(test,pa);
USE_GPU = true;
USE_CACHE = false;
USE_CAFFE = true;
for i=1:length(pos)
    pos(i).point=pos(i).point(1:length(pa),:);
end


% --------------------
% training

experiment_param.numPos = numPos;
experiment_param.numNeg = numNeg;
experiment_param.SVM_tol = SVM_tol;
experiment_param.SVM_C = SVM_C;
experiment_param.SSVM_epsilon = SSVM_epsilon;
experiment_param.HOG_coordinate_overlap = HOG_coordinate_overlap;
%traindate = date;
%trainname = sprintf('train_%s_%d_parts',traindate,tolpart);
%cnn_model = pycaffe_init_cnn_model();

%model = inittreemodel(name,pos,K,pa);
model = inittreemodel_Hogssvm(name,pos,K,pa);  
numLevels = model.interval;
%load trained_model
%for i=1:133  
%model.defs(i).anchor=trained_model.defs(i).anchor; 
%end
%trained_model = initmodel_Hogssvm_with_trained_filters(name,pos,K,pa,sbin);
%load('modelb4training_5_Hog.mat')
% wsol = model2vec(model);
% model = vec2model(ones(size(wsol)),model);

%load CNNSSVM_trainStart_18-Jul-2015_trainFinish_20-Jul-2015_3_parts_model_only
S=[];%empty structure to be inserted with positive data. 
Stest=[];
Strain=[];
 [Strain npos] = poslatent(Strain,name,1, model, pos);
% [Stest ntest] = poslatent(Stest,name,1, model, pos);
%[Strain npos] = pycaffe_poslatent(Strain,name, model, pos, experiment_param.numPos);
%[Stest ntest] = pycaffe_poslatent(Stest,name, model, test, numTest);



for phase = 1:1
    if phase == 1
        S = Strain;
        ndata = min(npos,numPos);
        %Pyras_train = [];
    else
        S = Stest;
        ndata = min(ntest,numTest);
        %Pyra_test = [];
    end

X=zeros(featdan,featdan,32,batch_size*numLevels);
y=zeros(batch_size*numLevels,12+4*tolpart);
ilcounter=0;
batch_counter = 0;
created_flag=false;
totalct=0;
  for i=1:ndata
    ilcounter = ilcounter +1;
%     [impyra, scales, level_sizes] = pycaffe_image_pyramid(S(i).x, cnn_model);
%     imsize = size(S(i).x);
%     pyra.scales = scales;  
%     pyra.level_sizes = level_sizes;
%     pyra.imsize = imsize(1:2); 
%     pyra.num_levels = cnn_model.pyra.num_levels;
%     pyra.stride = cnn_model.pyra.stride;
%     pyra.valid_levels = true(pyra.num_levels, 1);
%     pyra.padx = 0;
%     pyra.pady = 0;
%      pyra_boxes = im_to_pyra_coords(pyra, S(i).y);
      model.sbin=4;
      pyra     = featpyramid_Hogssvm(S(i).x,model);
      
      
      numLevels = length(pyra.scales);
        for l = 1:numLevels

      %  S(i).level{l}.y =  bsxfun(@times, S(i).y, pyra.scales(l));
      % boxlevel = bsxfun(@times, S(i).y, pyra.scales(l));
      %  S(i).level{l}.x = impyra(:,:,:,l);
        %vectorSi_y_ = pyra_boxes{l}';
        vectorSi_y_ = S(i).y';
        vectorSi_y = vectorSi_y_(:)';
        if size(pyra.feat{l},1) == 0 || size(pyra.feat{l},2) == 0
            break;
        end
        X(1:size(pyra.feat{l},2),1:size(pyra.feat{l},1),:,(ilcounter-1)*numLevels+l) = permute(pyra.feat{l},[2,1,3]);%S(i).level{l}.x;
       % y((ilcounter-1)*pyra.num_levels+l,:) =[vectorSi_y scales(l) l imsize(1) imsize(2) pyra.level_sizes(l,1) pyra.level_sizes(l,2) pyra.stride pyra.valid_levels(l) pyra.padx pyra.pady i phase];
        y((ilcounter-1)*numLevels+l,:) =[vectorSi_y pyra.scales(l) l  pyra.imy pyra.imx size(pyra.feat{l},1) size(pyra.feat{l},2) pyra.interval 1 pyra.padx pyra.pady i phase];
        
        fprintf('pyra.feat{%d} size [%d %d]\n',l,size(pyra.feat{l},1),size(pyra.feat{l},2));

        end
%         if i==1 
%             pause;
%         end
            if mod(i,batch_size) == 0          
            batch_counter = batch_counter +1;
            ilcounter = 0;
            if phase == 1
            name_save= sprintf('../examples/PARSE_ssvm26_lmdb/train_batch_mat/train_batch_%03d.mat',batch_counter);
            disp(name_save);
            else
            name_save= sprintf('../examples/PARSE_ssvm26_lmdb/test_batch_mat/test_batch_%03d.mat',batch_counter); 
            disp(name_save);
            end
           % disp(name_save);
            %data_x = X;
           % labels = y;
           % data = reshape(permute(X,[2,1,3,4]),[imdan*imdan*3,batch_size*cnn_model.pyra.num_levels])';
            %ydata(end+1:end+pyra.num_levels*batch_size,:) = y;
            %save(name_save,'data','labels');
           % dlabels = [labels data]';
              % to simulate maximum data to be held in memory before dumping to hdf5 file 
            batchdata=X; 
            batchlabs=y';
              % store to mat
            save(name_save,'batchdata','batchlabs');
            X=zeros(featdan,featdan,32,batch_size*numLevels);
            y=zeros(batch_size*numLevels,12+4*tolpart);
        end  
       
    % pyras{i} = pyra; 
  end

  if phase == 1   
    %save('PARSE/binlevel/ytrain','ydata');
    num_train_batches = batch_counter;
    %Pyras_train = pyras;
    
  else   
    %save('PARSE/binlevel/ytest','ydata');  
    num_test_batches = batch_counter;
    %Pyras_test = pyras;
  end

end
%h5disp('/working3/peerajak/ChulaQE/Semister8/9_caffe-master/examples/MySubgradientSSVMNN_softmax/hdf5_data/train.h5');
load modelpretrained_b4training5hog

save('../examples/PARSE_ssvm26_lmdb/ExpParamsHOGssvm26_lmdb_batch5','K','pa','experiment_param','model','modelpretrainedConvssvm32','modelb4training32','numPos','numNeg','numLevels','trained_model','numTest','batch_size','tolpart','num_train_batches','num_test_batches','featdan');
%save('pycaffe_Spos','S');
