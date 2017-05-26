function neg = INRIA_data
% neg = INRIA_data
% Grab list of negative training images from the INRIAPerson dataset
%  neg(i).im: image name for i-th negative image containing no human

files = dir('INRIA/*.jpg');

neg = [];
for i = 1:length(files),
  neg(i).im = ['INRIA/' files(i).name];
end
