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

'''
# These commented-out scripts are vestiges from the Arwen server, which are not applicable on the new server

# a child process (like this one) can't modify the environment variable
# so before running any python script that calls "os.system(cmd)",
# please make sure the environment variables are properly set.
source_dir_root = "/media/BednyDrobo/Projects/BSYN"
subjects_dir = "/media/BednyDrobo2/yunfei_reanalyze_BSYN"
extraction_dir = subjects_dir + "/extracted_data"

source_dir_subjects = [folder for folder in os.listdir(source_dir_root) if folder[:5]=="BSYN_"]
source_dir_subjects.sort()
source_CB_and_S = source_dir_subjects[:23]+source_dir_subjects[-21:]
# print(source_CB_and_S)

task = ["bsyn", "smath"]
run_dir = []
for tt in task:
	for ii in range(6):
		run_dir = run_dir + ["%s_%.2d.feat"%(tt,ii+1)]
# print(run_dir)
# run_dir = ["bsyn_01.feat", "bsyn_02.feat"..., "smath_06.feat"]

hemi = ["lh.32k_fs_LR.surfed_data.func.gii", "rh.32k_fs_LR.surfed_data.func.gii"]

gii_meta = {u'ProgramProvenance': u'Workbench\nVersion: Beta 0.85\nQt Compiled Version: 4.8.5\nQt Runtime Version: 4.8.4\ncommit: 14ba5c74f8cb92978ee1c0eb8b6a71de74c6479f\ncommit date: 2014-04-04 09:14:40 -0500\nCompiled Debug: NO\nOperating System: Linux', u'ParentProvenance': u'/media/BednyDrobo/Analyses/BSYN/SurfAnat/BSYN_CB_01/surf/lh.32k_fs_LR.midthickness.surf.gii:\n/usr/share/workbench/bin_linux64/../exe_linux64/wb_command -surface-resample /media/BednyDrobo/Analyses/BSYN/SurfAnat/BSYN_CB_01/surf/lh.midthickness.surf.gii /media/BednyDrobo/Analyses/BSYN/SurfAnat/BSYN_CB_01/surf/lh.sphere.reg.surf.gii /media/BednyDrobo/SurfAnat/32k_fs_LR/surf/lh.sphere.reg.surf.gii BARYCENTRIC /media/BednyDrobo/Analyses/BSYN/SurfAnat/BSYN_CB_01/surf/lh.32k_fs_LR.midthickness.surf.gii\n\nBSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.surfed_data.func.gii:\n/usr/share/workbench/bin_linux64/../exe_linux64/wb_command -metric-mask /tmp/preproc-mfU/lh.32k_fs_LR.surfed_data.func.gii BSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.mask.shape.gii BSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.surfed_data.func.gii\n\nBSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.mask.shape.gii:\n/usr/share/workbench/bin_linux64/../exe_linux64/wb_command -metric-math (mask*(abs(mw-1))) BSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.mask.shape.gii -fixnan 0 -var mask /tmp/preproc-mfU/lh.32k_fs_LR.mask.shape.gii -var mw /media/BednyDrobo/Analyses/BSYN/SurfAnat/32k_fs_LR/label/masks/lh.Medial_wall.shape.gii', u'Provenance': u'/usr/share/workbench/bin_linux64/../exe_linux64/wb_command -metric-smoothing /media/BednyDrobo/Analyses/BSYN/SurfAnat/BSYN_CB_01/surf/lh.32k_fs_LR.midthickness.surf.gii BSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.surfed_data.func.gii .84932181652950119329 BSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.surfed_data.func.gii -roi BSYN_CB_01/preproc/bsyn_01.feat/lh.32k_fs_LR.mask.shape.gii', u'AnatomicalStructurePrimary': u'CortexLeft', u'WorkingDirectory': u'/media/BednyDrobo/Analyses/BSYN'}

def try_to_read_gii():
	the_file = source_dir_root+'/'+source_CB_and_S[0]+'/preproc/bsyn_01.feat/lh.32k_fs_LR.surfed_data.func.gii'
	# print(the_file)
	data = nibg.read(the_file)
	darray1 = data.darrays[0]
	# print(len(darray1.data))
	return data

def save_array_to_gii():
	# 180922: this function is not completed yet.
	# 180922: can save to a gii file, and print_summary() shows the same result as an original gii file,
	# 180922: but hcp viewer can't recognize it.
	file_name = "%s/extracted_data/CB_01-bsyn_01-lh.npy"%subjects_dir
	data = np.load(file_name)
	data_good = try_to_read_gii()
	# data_good.print_summary()
	labtab = data_good.get_labeltable()

	# # test case added in 181008
	# # extract the original data, and save it back to .gii
	# data_arrays = data_good.darrays
	# dims = (len(data_arrays[0].data), len(data_arrays))
	# # print(dims)
	# extraction = np.empty(dims)
	# for ii, da in enumerate(data_arrays):
	# 	data[:,ii] = da.data
	data = data[:,0]
	if len(data.shape)==1:
		print("expand!!")
		data = data[:,np.newaxis]
	print(data.shape)

	arr_size = data.shape[1]
	darrays = []
	for ii in range(arr_size):
	
		darrays += [nibg.gifti.GiftiDataArray.from_array(data[:,ii])]
		# don't specify the datatype as FLOAT32
	# print(darrays)
	
	img = nibg.gifti.GiftiImage(darrays=darrays,header=data_good.header, labeltable=labtab, version="1")
	# print(img)
	
	# print(data)
	print()
	# img.print_summary()
	nibg.giftiio.write(img, "test_gifti.func.gii")
'''

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


