function MLPRep = getMedRepMLP (full, numbits, numfiles, repFileName)

data = load(repFileName);
allbits = 2000;
MLPRep = {};

if full == 'f'
    subBit = randperm(allbits);
    subBit = sort(subBit(1:numbits));
end

deltaDelay = 16;

for count = 1:numfiles
   curfield = strcat('n',int2str(count-1));
   curRep = getfield(data,curfield);
   curRep = curRep';
   med = median(curRep,2) * ones(1,size(curRep,2));
   
   MLPRep{count} = (curRep > med);
   
   if full == 'f'
       subRep = [];
       for i = 1:numbits
           subRep = cat(1,subRep,MLPRep{count}(subBit(i),:));
       end
       MLPRep{count} = subRep;
   end
end

end