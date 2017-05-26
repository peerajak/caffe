% Set up global variables used throughout the code
% addpath learning;
% addpath detection;
% addpath BUFFY/code/;
% addpath BUFFY/code/utils/;

% directory for caching models, intermediate data, and results
cachedir = 'cache/';
if ~exist(cachedir,'dir')
    unix(['mkdir ' cachedir]);
end

if ~exist([cachedir 'imflip/'],'dir'),
  unix(['mkdir ' cachedir 'imflip/']);
end
