function [im, box] = croppos(im, box)

  if iscell(box) % many parts are stored in box
    P = length(box);
    x1 = zeros(1,P);
    y1 = zeros(1,P);
    x2 = zeros(1,P);
    y2 = zeros(1,P);
    for p = 1:P
      x1(p) = box{p}(1);
      y1(p) = box{p}(2);
      x2(p) = box{p}(3);
      y2(p) = box{p}(4);
    end
    x1 = min(x1); y1 = min(y1); x2 = max(x2); y2 = max(y2);
    pad = 0.5*((x2-x1+1)+(y2-y1+1));
    x1 = max(1, round(x1-pad));
    y1 = max(1, round(y1-pad));
    x2 = min(size(im,2), round(x2+pad));
    y2 = min(size(im,1), round(y2+pad));
    
    im = im(y1:y2, x1:x2, :);
    for p = 1:P
      box{p}([1 3]) = box{p}([1 3]) - x1 + 1;
      box{p}([2 4]) = box{p}([2 4]) - y1 + 1;
    end
    
  else % only one template is stored in box
       % crop image around bounding box
    pad = 0.5*((box(3)-box(1)+1)+(box(4)-box(2)+1));
    x1 = max(1, round(box(1) - pad));
    y1 = max(1, round(box(2) - pad));
    x2 = min(size(im, 2), round(box(3) + pad));
    y2 = min(size(im, 1), round(box(4) + pad));
    
    im = im(y1:y2, x1:x2, :);
    box([1 3]) = box([1 3]) - x1 + 1;
    box([2 4]) = box([2 4]) - y1 + 1;
  end
  
  
  