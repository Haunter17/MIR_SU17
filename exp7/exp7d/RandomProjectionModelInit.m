function RandomProjectionModelInit(filelist, modelFile, parameter)
	
	%%%
	% Input: filelist, a list of reference files for an artist. In our case these 
	% won't be used (in the default model PCA was performed on them)
	%
	% Output: write any necessary information about the model to modelFile
	%%%
	if nargin<3
	    parameter=[];
	end
	if isfield(parameter,'m')==0
	    parameter.m=20;
	end
	if isfield(parameter,'tao')==0
	    parameter.tao=1;
	end
	if isfield(parameter,'hop')==0
	    parameter.hop=5;
	end
	if isfield(parameter,'deltaDelay')==0
	    parameter.deltaDelay=16;
	end
	if isfield(parameter,'precomputeCQT')==0
	    parameter.precomputeCQT = 0;
	end

	if isfield(parameter,'numFiltersList')==0
    	parameter.numFiltersList = [10, 20, 30, 40];
	end
	if isfield(parameter,'numFilterRows')==0
	    parameter.numFilterRows = 121;
	end
	if isfield(parameter,'numFilterCols')==0
	    parameter.numFilterCols = 483;
	end
	filterArray = {};
	% we want to make a cell array for the filters.
	curRows = parameter.numFilterRows;
	curCols = parameter.numFilterCols;
	for layerIndex = 1:length(parameter.numFiltersList)
		% each layer of filters is a 3-d matrix
		curFilters = zeros(curRows, curCols, parameter.numFiltersList(layerIndex));

		for filterIndex=1:parameter.numFiltersList(layerIndex)
			if (strcmp(parameter.FilterType,'RowMean'))
		    	curFilters(:, :, filterIndex) = createRandomFilter(curRows, curCols);
		    elseif(strcmp(parameter.FilterType,'FullMean'))
		    	curFilters(:, :, filterIndex) = createRandomFilterZeroFullMean(curRows, curCols);
		    end
		    %imagesc(filters(:,:,i));
		    
		end
		% store this layer of filters
		filterArray{layerIndex} = curFilters
		% we'll just be applying filters to a strung out output from this layer
		% where each filter in this layer gives one input to the next layer
		curRows = parameter.numFiltersList(layerIndex)
		curCols = 1
	end
	%% Save to file
	size(filterArray)
	parameter
	disp(['Saving filters to file']);
	save(modelFile,'filelist','parameter','filterArray');
end

function filter = createRandomFilter(numRows, numCols)
    % for now generate from a uniform random distribution
    % nx1 filters break this, since making them mean 0 --> it's just 0
    % then you try to normalize 0 and end up dividing by 0.
    filter = randn(numRows, numCols);
    % subtract off mean from each frequency channel to make volume
    % invariant
    rowMeans = mean(filter, 2);
    filter = bsxfun(@minus, filter, rowMeans);
    % normalize
    filter = filter ./ sqrt(sum(filter(:).^2));
end

function filter = createRandomFilterZeroFullMean(numRows, numCols)
    % for now generate from a uniform random distribution
    % 1x1 filters break this, since making them mean 0 --> it's just 0
    % then you try to normalize 0 and end up dividing by 0.
    filter = randn(numRows, numCols);
    % find the mean of the whole thing and subtract it off
    avg = mean(filter(:));
    filter = filter - avg;
    % normalize
    filter = filter ./ sqrt(sum(filter(:).^2));
end

% example call:
% x.numFiltersList = [256]
% RandomProjectionModelInit('', 'out', x)

