function reps = RandomProjectionModelGetReps(modelfile, maxPitchShift, filesToRep)
	%%%
	% Input:  modelfile: a file containing the filters for our model
	%         filelist: a list of files to generate representations for 
	% Output: Return a cell array. Each cell in the cell array is the representation for a given file
	%  	The representations are each a matrix of binary
	% 	values, with each column being the representation at a single point in time.
	%
	%%%

	model = load(modelfile); % loads filelist, parameter, and filterArray. In our case we didn't use filelist to make our filters
	parameter = model.parameter;
	filterArray = model.filterArray;

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
        logQspec = preprocessQspec(Q, parameter.DownsamplingRate);

        fprintf('Time for CQT: %f\n', toc(preQ));

        hashTic = tic;
        origfpseq = computeAlphaHashprints(logQspec, model, parameter);
        fprintf('Single hash takes: %f\n', toc(hashTic));
        % store all the pitch shifted versions
    	fpseqs = false(parameter.numFiltersList(end),size(origfpseq,2),2*maxPitchShift+1);
    	fpseqs(:,:,1) = origfpseq;

        % compute hashprints on pitch-shifted versions
        for i=1:maxPitchShift % shift up
    	    logQspec = preprocessQspec(pitchShiftCQT(Q,i), parameter.DownsamplingRate);
            fpseqs(:,:,i+1) = computeAlphaHashprints(...
                logQspec, model,parameter);
        end
        for i=1:maxPitchShift % shift down
    	    logQspec = preprocessQspec(pitchShiftCQT(Q,-1*i), parameter.DownsamplingRate);
            fpseqs(:,:,i+1+maxPitchShift) = computeAlphaHashprints(...
                logQspec,model,parameter);
        end
        
        fingerprints{count} = fpseqs;
    	idx2file{count} = curfile;
    	count = count + 1;
    	curfile = fgetl(fid);

	end
	reps = fingerprints;
end

function F = computeAlphaHashprints(spec,model,parameter)
	% if nargin < 3
	%     parameter=[];
	% end

	% numColSpec = size(spec, 2);
	% [~, numColFilter, numFilters] = size(filters);
	% numColFeatures = numColSpec - numColFilter + 1;
	% features = zeros(numFilters, numColFeatures);

	filterArray = model.filterArray;

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
		% deal with issues if there is no third dimension in filters
		if (numFilters == 1)
			layerOutput(1, :) = conv2(filterInput, filters, 'valid'); % for now assume leads to a row vector from the conv (filter has the same height as its input)
		else
			for filterIndex = 1:numFilters
				filt = filters(:, :, filterIndex);
				layerOutput(filterIndex, :) = conv2(filterInput, filt, 'valid'); % for now assume leads to a row vector from the conv (filter has the same height as its input)
			end
		end
		% apply the non-linearity except in the last layer
		if layerIndex < length(filterArray)
			if (strcmp(parameter.nonlinearity, 'relu'))
				layerOutput = relu(layerOutput);
			elseif (strcmp(parameter.nonlinearity, 'sigmoid'))
				layerOutput = sigmoid(layerOutput);
			end
		end
		filterInput = layerOutput;
	end

	% deal with delta features if need be
	if (parameter.DeltaFeatures)
		layerOutput = layerOutput(:, 1:(size(layerOutput, 2) - parameter.DeltaDelay)) - layerOutput(:, (1 + parameter.DeltaDelay):end);
	else

	end
	% threshold the final layer's output for each column by its median and return
	% do this by subtracting off the median then thresholding at 0
	if (strcmp(parameter.ThresholdStrategy, 'zero'))
		F = layerOutput > 0;

	elseif (strcmp(parameter.ThresholdStrategy, 'median'))
		F = (bsxfun(@minus, layerOutput, median(layerOutput, 2))) > 0;
	end


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

function y = relu(x)
	y = x;
	y(y < 0) = 0;
end

function y = sigmoid(x)
	y = 1./(1 + exp(-1 * x));
end



