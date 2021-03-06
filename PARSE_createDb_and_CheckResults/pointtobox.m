function pos = pointtobox(pos,pa)

len = zeros(length(pos),length(pa)-1);
for n = 1:length(pos)
  points = pos(n).point;
  for p = 2:length(pa)
    len(n,p-1) = norm(abs(points(p,1:2)-points(pa(p),1:2)));
  end
end

r = zeros(1,length(pa)-1);
for i = 1:length(pa)-1
    ratio = log(len(:,i))-log(len(:,1));
    r(i) = exp(median(ratio));
end

% r = [1	0.678778462194631	0.565993047561248	0.565993047561248	0.582518524287417	0.582518524287417	0.716659226014266	0.716659226014266	0.716659226014267	0.821534619055122	0.821534619055122	0.854977479719453	0.854977479719453	0.678778462194631	0.565993047561248	0.565993047561248	0.582518524287417	0.582518524287417	0.716659226014266	0.716659226014266	0.716659226014267	0.821534619055122	0.821534619055122	0.854977479719453	0.854977479719453];


boxsize = zeros(1,length(pos));
for n = 1:length(pos)
  ratio = len(n,:)./r;
  boxsize(n) = quantile(ratio,0.85);
end

for n = 1:length(pos)
    points = pos(n).point;
    for p = 1:length(pa)
      pos(n).x1(p) = points(p,1) - boxsize(n)/2;
      pos(n).y1(p) = points(p,2) - boxsize(n)/2;
      pos(n).x2(p) = points(p,1) + boxsize(n)/2;
      pos(n).y2(p) = points(p,2) + boxsize(n)/2;
    end

show = false;
  if n<5 && show
        figure ; clf;
        im = imread(pos(n).im);
        imshow(im);hold on;
        for p = 1:length(pa)
            rectangle('Position',[pos(n).x1(p),pos(n).y1(p),pos(n).x2(p)-pos(n).x1(p),pos(n).y2(p)-pos(n).y1(p)],'EdgeColor','b','linewidth',2)
            text(points(p,1),points(p,2),num2str(p));
        end
        drawnow;
        figure(2); clf;
        imshow(im);hold on;
        for p = 1:length(pa)
            plot(points(p,1),points(p,2),'b.','markersize',14);
            text(points(p,1),points(p,2),num2str(p));
        end
        drawnow;
    end

end
