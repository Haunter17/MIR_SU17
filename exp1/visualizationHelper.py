
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def visualizeWeights(listOfWeightMatrices, numRows, outfile):
	numMatrices = len(listOfWeightMatrices)
	numCols = math.ceil(float(numMatrices) / numRows) # so that don't end up with an extra row

	# make a figure with subplots
	fig = plt.figure(figsize=(3 * numCols, 4 * numRows))

	for index in range(numMatrices):
		curMatrix = listOfWeightMatrices[index]
		subplot = fig.add_subplot(numRows, numCols, index+1)
		subplot.matshow(curMatrix)

	fig.tight_layout()
	fig.savefig(outfile)

'''
Visualize each column from the matrix - split the figure into numRows numRows
# assume taking in a numpy matrix
'''
def visualizeColVecs(matrix, numRows, outfile):

	matplotlib.rcParams.update({'font.size': 8})


	numColsInMatrix = matrix.shape[1]
	numCols = math.ceil(float(numColsInMatrix) / numRows) # so that don't end up with an extra row

	# make a figure with subplots
	fig = plt.figure(figsize=(1 * numCols, 1 * numRows))
	#fig.wspace = 0
	#fig.hspace = 0
	#gs1 = gridspec.GridSpec(numRows, numCols)
	#gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

	# iterate over each column - transpose it to get the columns
	index = 0
	for column in matrix.T:
		subplot = fig.add_subplot(numRows, numCols, index+1)
		#subplot.set_aspect('equal')
		#subplot.matshow([[i] for i in column]) # make it into a column vector instead of a row vector by nesting each element in a list
		subplot.plot(range(len(column)), column) # instead plot value vs. frequency
		subplot.set_xticks([])
		subplot.set_yticks([])
		#subplot.axis('off')
		index = index + 1
	
	#fig.subplots_adjust(wspace=None, hspace=None)
	fig.tight_layout()
	fig.savefig(outfile)



