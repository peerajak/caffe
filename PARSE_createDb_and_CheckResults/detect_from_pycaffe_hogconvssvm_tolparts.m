function result_dirname = detect_from_pycaffe_hogconvssvm_tolparts(feat_prototxt,feat_caffemodel,convssvm_mat,ini_model_mat,Experiment_name)

%1. load .mat file. This should gives tolpart,bias_def_vector
%,convssvm_w_matrix,model,thresh
load(convssvm_mat);
load(ini_model_mat);
convssvm = permute(convssvm,[3 4 2 1]);
w = create_wsol_from_convssvm_w_matrix(tolpart,lossw ,convssvm,model);
load trained_model
model = vec2model(w,model);
for ij = 1:133
    model.defs(ij).anchor = trained_model.defs(ij).anchor;
end
model.thresh = -10;%thresh;
USE_GPU = true;
USE_CACHE = false;
USE_CAFFE = true;
cnn_feat_model = init_cnn_model_anyfiles100('use_gpu', USE_GPU, 'use_caffe', USE_CAFFE,'def_file',feat_prototxt,'net_file', feat_caffemodel);
model.interval = cnn_feat_model.pyra.num_levels;
result_dirname = ['result_nopadding_' Experiment_name];
result_trainData_name = [result_dirname '/trainData'];
result_testData_name = [result_dirname '/testData'];
mkdir(result_trainData_name);
mkdir([result_trainData_name '/PARSE']);
mkdir(result_testData_name);
mkdir([result_testData_name '/PARSE']);
globals;
name = 'PARSE';
[pos test] = PARSE_data14();
neg        = INRIA_data();


PCP_CAL = false;
if PCP_CAL
suffix = num2str(K')';
[boxes_test points_test] = testmodelHogConvPrecision(name,model,cnn_feat_model,test,suffix);
save('points_test','points_test','name','test');
[detRate_test PCP_test R_test] = PARSE_eval_pcp(name,points_test,test);
fprintf('detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate_test,PCP_test,detRate_test*PCP_test);
[boxes_train points_train] = testmodelHogConv(name,model,cnn_feat_model,pos,suffix);
[detRate_train PCP_train R_train] = PARSE_eval_pcp(name,points_train,pos);
fprintf('detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate_train,PCP_train,detRate_train*PCP_train);
save([cachedir name '_pcp_' result_dirname],'detRate_test','PCP_test','R_test','detRate_train','PCP_train','R_train','boxes_test','boxes_train');
end


[pos test] = PARSE_data();
test = pointtobox(test,pa);
pos  = pointtobox(pos,pa);
% for i=1:100
%     im = imread(pos(i).im);
%     %im = imresize(im,1.8);
%    %  figure;
%   %  clf; imagesc(im); axis image; axis off; drawnow;
%     fprintf('detecting training %s\n',pos(i).im);
%     boxes = detectHogconv(im, model, cnn_feat_model,min(model.thresh,0));  
%     boxes = nms(boxes, .1); % nonmaximal suppression
%   %  showboxes(im, boxes(1,:),colorset); % show the best detection
%     
%     if ~isempty(boxes)
%     outimfilename = sprintf('%s/%s',result_trainData_name ,pos(i).im);   
%     im_box_print_color(im, tolpart, boxes(1,:),outimfilename);
%     end
% end
for i=1:100
    im = imread(test(i).im);
   % im = imresize(im,1.1);
   %  figure;
  %  clf; imagesc(im); axis image; axis off; drawnow;
    boxes = detectHogconv(im, model, cnn_feat_model, min(model.thresh,0));   
    boxes = nms(boxes, .1); % nonmaximal suppression
  %  showboxes(im, boxes(1,:),colorset); % show the best detection
    
    if ~isempty(boxes)
    outimfilename = sprintf('%s/%s',result_testData_name ,test(i).im);   
    im_box_print_color(im, tolpart, boxes(1,:),outimfilename);
    end
end
%caffe('reset');
