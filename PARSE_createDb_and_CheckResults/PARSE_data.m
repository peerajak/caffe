function [pos test] = PARSE_data
% this function is very dataset specific, you need to modify the code if
% you want to apply the pose algorithm on some other dataset

% it converts the various data format of different dataset into unique
% format for pose detection 
% the unique format for pose detection contains below data structure
%   pos:
%     pos(i).im: filename for the image containing i-th human 
%     pos(i).point: pose keypoints for the i-th human
%   test:
%     test(i).im: filename for i-th testing image
% This function also prepares flipped images for training.

globals;

load PARSE/labels.mat;
posims = 'PARSE/im%.4d.jpg';

trainfrs = 1:100; % training frames
testfrs = 101:305; % testing frames

% -------------------
% grab positive annotation and image information
pos = [];
numpos = 0;
for fr = trainfrs
  numpos = numpos + 1;
  pos(numpos).im = sprintf(posims,fr);
  pos(numpos).point = ptsAll(:,:,fr);
end

% -------------------
% flip positive training images
posflipims = [cachedir 'imflip/PARSE%.6d.jpg'];
for n = 1:length(pos)
  if exist(sprintf(posflipims,n),'file')
    continue;
  end
  im = imread(pos(n).im);
  im_flip = im(:,end:-1:1,:);
  imwrite(im_flip,sprintf(posflipims,n));
end

% -------------------
% flip labels for the flipped positive training images
% mirror property for the keypoint, please check your annotation for your
% own dataset
mirror = [6 5 4 3 2 1 12 11 10 9 8 7 13 14];
for n = 1:length(pos)
  im = imread(pos(n).im);
  width = size(im,2);
  numpos = numpos + 1;
  pos(numpos).im = sprintf(posflipims,n);
  for p = 1:size(pos(n).point,1)
    pos(numpos).point(mirror(p),1) = width - pos(n).point(p,1) + 1;
    pos(numpos).point(mirror(p),2) = pos(n).point(p,2);
  end
end

% -------------------
% Create ground truth keypoints for model training
% We augment the original 14 joint positions with midpoints of joints, 
% defining a total of 26 keypoints
I = [1  2  3  4   4   5  6   6   7  8   8   9   9   10 11  11  12 13  13  14 ...
           15 16  16  17 18  18  19 20  20  21  21  22 23  23  24 25  25  26];
J = [14 13 9  9   8   8  8   7   7  9   3   9   3   3  3   2   2  2   1   1 ...
           10 10  11  11 11  12  12 10  4   10  4   4  4   5   5  5   6   6];
S = [1  1  1  1/2 1/2 1  1/2 1/2 1  2/3 1/3 1/3 2/3 1  1/2 1/2 1  1/2 1/2 1 ...
           1  1/2 1/2 1  1/2 1/2 1  2/3 1/3 1/3 2/3 1  1/2 1/2 1  1/2 1/2 1];
A = full(sparse(I,J,S,26,14));

for n = 1:length(pos)
  pos(n).point = A * pos(n).point; % linear combination
end

% -------------------
% grab testing image information
test = [];
numtest = 0;
for fr = testfrs
  numtest = numtest + 1;
  test(numtest).im = sprintf(posims,fr);
  test(numtest).point = ptsAll(:,:,fr);
end
for n = 1:length(test)
  test(n).point = A * test(n).point; % linear combination
end