function [S numpositives] = poslatent(S,name, t, model, pos)
  
  numpos = length(pos);
  numpositives = zeros(length(model.components), 1);
  minsize = prod(model.maxsize);
  
  % we add the below lines 
  numparts = length(model.components{1});
  
  for i = 1:numpos

    fprintf('%s: iter %d: latent positive: %d/%d\n', name, t, i, numpos);
    % skip small examples
    skipflag = 0;
    bbox = cell(1,numparts);
    for p = 1:numparts
      bbox{p} = [pos(i).x1(p) pos(i).y1(p) pos(i).x2(p) pos(i).y2(p)];
      if (bbox{p}(3)-bbox{p}(1)+1)*(bbox{p}(4)-bbox{p}(2)+1) < minsize
        skipflag = 1;
        break;
      end
    end
    if skipflag
      continue;
    end
    
    % get example
    im = imread(pos(i).im);
    if size(im,1)>300
        im = imresize(im,[360, NaN]);
    end
    if size(im,2)>300
        im = imresize(im,[NaN,460]);
    end
    [im, bbox] = croppos(im, bbox);
    sizeimx = size(im,2);
    sizeimy = size(im,1);
    sizebox =30;
    marginleft = 10;
    randx1 = (sizeimx-sizebox-marginleft)*rand(26,1);
    randy1 = (sizeimy-sizebox-marginleft)*rand(26,1);
     S(end+1).x = im;
        for iii=1:numparts            
        S(end).y(iii,:) = bbox{iii};
         end
         S(end).yhat =[randx1 randy1 randx1+sizebox randy1+sizebox];      
         S(end).ytype = 1;
    %     S(1).y
   % figure; imshow (im); pause;
%     box = detect(im, model, 0);
%       save('cropposeExample','im','bbox','box');
%     if ~isempty(box),
%       fprintf(' (comp=%d,sc=%.3f)\n',box(1,end-1),box(1,end));
%       c = box(1,end-1);
%       numpositives(c) = numpositives(c)+1;
%       %      showboxes(im, box);
%     end
   end
  numpositives = i;

