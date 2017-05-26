function w = create_wsol_from_convssvm_w_matrix(tolpart,bias_def_vector ,convssvm_w_matrix,model)
% [w,wreg,w0,nonneg] = model2vec(model)

w     = zeros(model.len,1);
fprintf('model length for w is %d\n',model.len);

%filters_cnn_model_matrix = permute(cnn_model_Loss.net.params('convssvm', 1).get_data(),[3 4 2 1]);
filters_cnn_model_matrix = convssvm_w_matrix;
size(bias_def_vector)
for x = model.bias
  j = x.i:x.i+numel(x.w)-1;
  fprintf('j from %d to %d\n',min(j),max(j));  
  w(j) = bias_def_vector(j);
end
 %filters
 
for ipart =1:tolpart
    for x = model.components{1}(ipart).filterid,   
      fprintf('x filterid = %d\n',x);
      j = model.filters(x).i:model.filters(x).i+numel(model.filters(x).w)-1;
      fprintf('j from %d to %d\n',min(j),max(j));
      w(j) = double(reshape(filters_cnn_model_matrix(:,:,:,x),[numel(model.filters(x).w),1]));
    end  
end

for x = model.defs
  j = x.i:x.i+numel(x.w)-1;
  w(j) = bias_def_vector(j);
  % Enforce minimum quadratic deformation costs of .01


end

