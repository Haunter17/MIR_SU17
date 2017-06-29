function reps = RandomProjectionModelGetReps(modelfile, maxPitchShift, filesToRep)
	%%%
	% Input:  modelfile: a file containing the filters for our model
	%         filelist: a list of files to generate representations for 
	% Output: Return a cell array. Each cell in the cell array is the representation for a given file
	%  	The representations are each a matrix of binary
	% 	values, with each column being the representation at a single point in time.
	%
	%%%

	load(modelfile); % loads filelist, parameter, and filterArray. In our case we didn't use filelist to make our filters
	fingerprints = {};
	idx2file = {};

	% iterate through the files we want to represent
	fid = fopen(filesToRep);
	count = 1;
	curfile = fgetl(fid);
	while ischar(curfile)
		preQ = tic;
    	disp(['==> Computing fingerprints on file ',num2str(count),': ',curfile]);
        
        Q = computeQSpec(curfile,parameter);
        logQspec = preprocessQspec(Q);

        fprintf('Time for CQT: %f\n', toc(preQ));

        hashTic = tic;
        origfpseq = computeAlphaHashprints(logQspec, filterArray, parameter);
        fprintf('Single hash takes: %f\n', toc(hashTic));
        % store all the pitch shifted versions
    	fpseqs = false(parameter.numFiltersList(end),size(origfpseq,2),2*maxPitchShift+1);
    	fpseqs(:,:,1) = origfpseq;

        % compute hashprints on pitch-shifted versions
        for i=1:maxPitchShift % shift up
    	    logQspec = preprocessQspec(pitchShiftCQT(Q,i));
            fpseqs(:,:,i+1) = computeAlphaHashprints(...
                logQspec,filterArray ,parameter);
        end
        for i=1:maxPitchShift % shift down
    	    logQspec = preprocessQspec(pitchShiftCQT(Q,-1*i));
            fpseqs(:,:,i+1+maxPitchShift) = computeAlphaHashprints(...
                logQspec,filterArray,parameter);
        end
        
        fingerprints{count} = fpseqs;
    	idx2file{count} = curfile;
    	count = count + 1;
    	curfile = fgetl(fid);

	end
	reps = fingerprints;
end

function F = computeAlphaHashprints(spec,filterArray,parameter)
	% if nargin < 3
	%     parameter=[];
	% end

	% numColSpec = size(spec, 2);
	% [~, numColFilter, numFilters] = size(filters);
	% numColFeatures = numColSpec - numColFilter + 1;
	% features = zeros(numFilters, numColFeatures);

	filterInput = spec;
	for layerIndex = 1:length(filterArray)
		% get info about the input to the filter
		filterInputRows = size(filterInput, 1);
		filterInputCols = size(filterInput, 2);
		% get this layer's filters
		filters = filterArray{layerIndex};
		filterRows = size(filters, 1);
		filterCols = size(filters, 2);
		numFilters = size(filters, 3);
		% loop through each filter and add it to your representation
		numColInput = size(filterInput, 2);
		numColOutput =  filterInputCols - filterCols + 1;
		layerOutput = zeros(numFilters, numColOutput);
		for filterIndex = 1:numFilters
			filt = filters(:, :, filterIndex);
			layerOutput(filterIndex, :) = conv2(filterInput, filt, 'valid'); % for now assume leads to a row vector from the conv (filter has the same height as its input)
		end
		filterInput = layerOutput;
	end
	% threshold the final layer's output at 0 and return
	F = layerOutput > 0;


	% way of thinking about it generally as convolutions - not sure how to get this
	% to work out though, going to do it under assumptions now

	% % F = features > 0;
	% numColSpec = size(spec, 2);

	% % the dimension of whats getting input to the filter
	% filterInput = spec;

	% for layerIndex = 1:length(filterArray)
	% 	% get info about the input to the filter
	% 	filterInputRows = size(filterInput, 1);
	% 	filterInputCols = size(filterInput, 2);
	% 	% get info about the current filter
	% 	filters = filterArray{layerIndex}
	% 	filterRows = size(filters, 1);
	% 	filterCols = size(filters, 2);
	% 	numFilters = size(filters, 3);
	% 	% find the dimensions of what will result from filtering
	% 	layerOutputRows = filterInputRows - filterRows + 1;
	% 	layerOutputCols = filterInputCols - filterCols + 1;

	% 	layerOutput = zeros(layerOutputRows, layerOutputCols, numFilters)

	% 	for filterIndex = 1:numFilters
	% 		filt = filters(:, :, filterIndex)
			
	% 	end	

	% 	% update the dimension of what's being filtered in the next layer
	% 	filterInputRows = featureRows
	% 	filterInputCols = featureCols
	% end

	% assume the filters are the size of the input

end




