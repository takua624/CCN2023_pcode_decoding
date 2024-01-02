import os
import numpy as np
import nibabel.gifti as nibg
import nibabel as nib
import itertools
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import math
import pickle



def extract_gii(file_name):
	'''Extract a numpy array from .gii file
	
	Argument:
	file_name -- file name of the .gii file. It's a string, of course
	
	output:
	extraction -- a numpy array, dimensions = (vertice, TR)
	'''
	if file_name.split(".")[-1]=="mgh":
		extraction = np.array(nib.load(file_name).get_data())[:,:,0].astype("float64")
	if file_name.split(".")[-1]=="gii":
		data = nibg.read(file_name)
		data_arrays = data.darrays
		dims = (len(data_arrays[0].data), len(data_arrays))
		extraction = np.empty(dims)
		for ii, da in enumerate(data_arrays):
			extraction[:,ii] = da.data
	return extraction

def save_gii(data, filename, detailed=False):
	'''Save a numpy array to .gii format

	Arguments:
	data -- the data array (dimensions: (vertice, TR))
	filename -- file name (specify .func here)
	'''
	if len(data.shape)==1:
		data = data[:,np.newaxis]
	# If it's a brain map, there is only one dimension.
	# We have to specify that the size of the second dimension as 1
	data = data.astype("float32")
	arr_size = data.shape[1]
	darrays = []
	for ii in range(arr_size):
		darrays += [nibg.gifti.GiftiDataArray.from_array(data[:,ii])]
	
	if detailed:
		template = nibg.read("/export/bedny/Projects/TWFA/Group_main/lh_3s_no_derivative.rfx/cope32/con1/sig_fdr.func.gii")
		img = nibg.gifti.GiftiImage(darrays=darrays, header=template.header, file_map=template.file_map, meta=template.meta, labeltable=template.labeltable, extra=template.extra)
	else:
		img = nibg.gifti.GiftiImage(darrays=darrays)
	nibg.giftiio.write(img, filename)
	return


