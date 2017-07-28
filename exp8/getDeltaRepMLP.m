function MLPRep = getDeltaRepMLP (full, numbits, numfiles)

data = load('exp8_Representations/exp8a_new/exp8a_relu_repNon.mat');
MLPRep = {};

if full == 'f'
    subBit = randperm(4000);
    subBit = sort(subBit(1:numbits));
end

deltaDelay = 16;

for count = 1:numfiles
   curfield = strcat('n',int2str(count-1));
   curRep = getfield(data,curfield);
   curRep = curRep';
   deltas = curRep(:,1:(size(curRep,2)-deltaDelay)) - curRep(:,(1+deltaDelay):end);
   
   MLPRep{count} = (deltas > 0);
   
   if full == 'f'
       subRep = [];
       for i = 1:numbits
           subRep = cat(1,subRep,MLPRep{count}(subBit(i),:));
       end
       MLPRep{count} = subRep;
   end
end

end