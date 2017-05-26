function model = inittreemodel_Hogssvm(name,pos,K,pa)
  
numpart = length(pa); %each mix
part_model = cell(1,numpart);
for p = 1:numpart

  amodel = initmodel_Hogssvm(pos);
    models = cell(1,K(p));
    for k = 1:K(p)     
      models{k} = amodel;
    end
    part_model{p} = mergemodels(models);
    part_model{p}.name =p;
end
def = data_def(pos,amodel);
idx = clusterparts(def,K,pa);
ourdef=def;
ouridx=idx;
save('ourdefidx','ourdef','ouridx');
load PARSE_cluster_55566665555555566665555555
model = buildmodel(name,part_model,pa,def,idx,K);
