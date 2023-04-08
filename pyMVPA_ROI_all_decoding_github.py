import os
import pandas as pd
import numpy as np
from save_gii import extract_gii
from save_gii import save_gii
from random import sample
from scipy.stats import ttest_1samp, ttest_rel
from datetime import datetime
from multiprocessing import Process
import time
from numpy.matlib import repmat
import mvpa2
from mvpa2.datasets.base import *
from mvpa2.tutorial_suite import dataset_wizard
from mvpa2.mappers.zscore import ZScoreMapper
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.svm import LinearCSVMC, LinearNuSVMC
from mvpa2.measures.base import CrossValidation
import pickle
import warnings

warnings.filterwarnings("ignore")

n_vtx = 32492
now_str = datetime.now().strftime("%Y%m%d%H%M%S")


def beta_patterns(roi_mask, subj, hemi, top_n=400, train_on_test_on={"train":(),"test":()}, sig_file="", label_to_id={}, individual_scaling=False, per_run_scaling=False, grand_normalization=False):
	# train_on_test_on: the values corresponding to "train" and "test" can be ("PRE", "text"), ("POST", "text"), ("POST", "code")
	# don't use this script to train and test on the same set of data!
	# sig_file: an example is "../%(subj)s_POST/firstlevel/pcodepost.%(hemi)s.ffx/zstat9.func.gii"
	# label_to_id: example: {"textrealFOR":1, "textrealIF":2, "coderealFOR":1, "coderealIF":2, "textfakeFOR":0, "textfakeIF":0, "codefakeFOR":0, "codefakeIF":0}
	
	
	roi_data_x = {"train":[], "test":[]}
	roi_data_y = {"train":[], "test":[]}
	run_info = {"train":[], "test":[]}
	datasets = {"train":None, "test":None}
	
	if top_n == -1:
		print("Actually, use %d vertices!!"%np.sum(roi_mask))
	
	top_idx = []
	for tt in ["train","test"]:
		if train_on_test_on["train"] == train_on_test_on["test"] and tt=="test":
			#print("Train and test on the same data, should use crossvalidation later!")
			continue
		
		prepost = train_on_test_on[tt][0]
		code_text = train_on_test_on[tt][1]
		this_subj = subj
		
		if tt=="train":
			sig_data = np.zeros((n_vtx,1))
			
			sig_file = sig_file%{"subj":this_subj,"hemi":hemi}
			
			sig_data = extract_gii(sig_file)
			sig_data[roi_mask==0]=-99
			good_vtx = int(np.sum(sig_data>-99))
				
			sig_data = np.reshape(sig_data,(1,-1))
			fixed_good_vtx = int(np.sum(sig_data>-99))
			select = min(top_n,int(np.sum(roi_mask)),int(np.sum(sig_data>0)))
			select = min(top_n,int(np.sum(roi_mask)))
			top_idx = list(sig_data.argsort()[0][-select:])
		
		roster = pd.read_csv("item_to_pe_%s.csv"%prepost)
		
		for ii,row in roster.iterrows():
			
			this_label = "".join([cc for cc in row["label"] if cc.isalpha()])
			if row["subj"] != this_subj:
				continue
			if this_label[:4] != code_text:
				continue
			if label_to_id[this_label]==0:
				continue
			
			this_file = "../%(subj)s_%(prepost)s/firstlevel_each_func/%(run)s.%(hemi)s.glm/stats/pe%(pe)d.func.gii"%{"subj":row["subj"], "run":row["run"], "hemi":hemi, "pe":row["pe"], "prepost":prepost}
			this_label = "".join([cc for cc in row["label"] if cc.isalpha()])
			
			
			#print("[%s] The current data is from: %s, %s"%(tt,prepost,row["label"]))
			this_data = extract_gii(this_file)
			roi_data_x[tt] += [list(this_data[top_idx,0])]
			roi_data_y[tt] += [label_to_id[this_label]]
			run_info[tt] += [int(row["run"].split("_")[-1])]
	
		roi_data_x[tt] = np.array(roi_data_x[tt])
		roi_data_y[tt] = np.array(roi_data_y[tt])
		run_info[tt] = np.array(run_info[tt])
		
		if grand_normalization:
			# since the data values had better be centered around 0,
			# even if we're not doing any fancy normalization to remove noise,
			# we should still subtract the mean from all data points and divide by the SD of all data points (across all samples and all features)
			this_mean = np.mean(roi_data_x[tt])
			this_std = np.std(roi_data_x[tt])
			roi_data_x[tt] = (roi_data_x[tt]-this_mean)/this_std
			
		datasets[tt] = dataset_wizard(roi_data_x[tt])
		datasets[tt].sa["targets"] = roi_data_y[tt]
		datasets[tt].sa["run"] = run_info[tt]
		
		if per_run_scaling:
			# normalize each feature in each run
			zmapper = ZScoreMapper(chunks_attr="run")
			zmapper.train(datasets[tt])
			datasets[tt] = datasets[tt].get_mapped(zmapper)
			
	# example for selecting a run of data from a dataset:
	# sub_ds = datasets["train"][{"run":[2]},:]
	return (datasets)

def do_SVM(datasets, bootstrap=0, proportion_data_used=1):
	# datasets: {"train":a_dataset, "test":a_dataset}
	acc = np.zeros((bootstrap+1))
	for bs in range(bootstrap+1):
		if datasets["test"]==None:
			# meaning training and testing on the same stuff, have to do cross validation
			nfold = NFoldPartitioner(attr="run")
			svm = LinearCSVMC()
			cv = CrossValidation(svm, nfold, errorfx=lambda p, t: np.mean(p == t))
			cv_results = cv(datasets["train"])
			acc[bs] = np.mean(cv_results)
			# let's consider bootstrapping later
		if datasets["test"]!=None:
			tt_acc = []
			for LOrun in np.unique(datasets["train"].sa.run):
				train_data = datasets["train"][datasets["train"].sa.run!=LOrun,:]
				test_data = datasets["test"][datasets["test"].sa.run==LOrun,:]
				svm = LinearCSVMC()
				svm.train(train_data)
				results = svm.predict(test_data.samples)
				tt_acc += [np.mean(results==test_data.sa.targets)]
			acc[bs] = np.mean(tt_acc)
	return acc

def main_ROI_MVPA(subj="", roi_list=[], bootstrap=1000, top_n=400,  label="", proportion_data_used=1, train_on_test_on={"train":(),"test":()},  sig_file_list=[], label_to_id={}, individual_scaling=False, per_run_scaling=False, grand_normalization=False):
	for roi,sig_file in zip(roi_list,sig_file_list):
		hemi = "lh"
		result_file = (label.split("/")[0]+"/")*(len(label.split("/"))>1)+roi.split("/")[-1].split(".")[0]+"_"+label.split("/")[-1]+".pickle"
		
		roi_mask = extract_gii(roi)
		datasets = beta_patterns(roi_mask, subj, hemi, top_n=top_n, train_on_test_on=train_on_test_on, sig_file=sig_file, label_to_id=label_to_id, per_run_scaling=per_run_scaling, grand_normalization=grand_normalization)
		
		acc = do_SVM(datasets, bootstrap=bootstrap, proportion_data_used=proportion_data_used)
		print("[%s] %s\n%s: %f"%(roi,str(train_on_test_on),subj,acc[0]))
		
		with open(result_file, "wb") as pkl:
			pickle.dump(acc,pkl)
	return

# this is used to check observed accuracy, 
def check_result(subjs=[], roi_list=[], label="", result_label=""):
	df = pd.DataFrame(columns=["subj"]+[roi.split("/")[-1].split(".")[0] for roi in roi_list])
	for ss in subjs:
		to_add = {"subj":ss}
		this_label = label%ss
		for roi in roi_list:
			result_file = (this_label.split("/")[0]+"/")*(len(this_label.split("/"))>1)+roi.split("/")[-1].split(".")[0]+"_"+this_label.split("/")[-1]+".pickle"
			f = open(result_file, "rb")
			acc = pickle.load(f)
			to_add[roi.split("/")[-1].split(".")[0]] = acc[0]
		df = df.append(to_add, ignore_index=True)
	df.to_excel(result_label+".xlsx", index=False)
	return

# PRE,POST pcode, code within condition decoding
# POST code<->pcode cross decoding
# PRE-POST cross decoding
top_n = 500
bootstrap = 0
proportion_data_used = 1
individual_scaling = False
per_run_scaling = True
grand_normalization = False
label_to_id = {"textrealFOR":1, "textrealIF":2, "coderealFOR":1, "coderealIF":2, "textfakeFOR":0, "textfakeIF":0, "codefakeFOR":0, "codefakeIF":0}



# PRE,POST pcode, code within condition decoding
# for the submission of CCN2023, the "only_POST_subjs" are irrelevant. Their experiment paradigm is different from the ones that have both PRE and POST data
# preprocessed neuroimaging data are available upon request
processes = []
top_n = 350
for prepost in ["PRE","POST"]:
	for cond in ["text"]+["code"]*(prepost=="POST"):
		train_on_test_on={"train":(prepost,cond),"test":(prepost,cond)}
		# cfd_parcel400_roi1 is IPS, cfd_parcel400_roi3 is PFC
		roi_list = ["cfd_parcel400_roi1.lh.gii","cfd_parcel400_roi3.lh.gii", "A1.lh.gii"]
		sig_file_list = ["../%(subj)s_%(prepost)s/firstlevel/pcode%(prepost_low)s.%(hemi)s.ffx/zstat%(stat)d.func.gii"%{"subj":"%(subj)s", "hemi":"%(hemi)s", "stat":1+8*(cond=="code"), "prepost":prepost, "prepost_low":prepost.lower()} for roi in roi_list] # creal>fake, don't use _each_func data to avoid circular definition
		
		label = "pyMVPA_pool_%s/%s[%dvtx]%s_%s"%(prepost,cond,top_n, "_grandSc"*grand_normalization,"%s")
		only_POST_subjs = ["NV_%.2d"%ii for ii in [4,12,13,15,17,18,25]]
		subjs = ["NV_%.2d"%ii for ii in [27,28,29,30,38,39,44,50,52,53,56,57,59,60,61,62,65,77]] + only_POST_subjs*(prepost=="POST")
		processes += [Process(target=main_ROI_MVPA, args=(subj, roi_list, bootstrap, top_n,  label%subj, proportion_data_used, train_on_test_on,  sig_file_list, label_to_id, individual_scaling, per_run_scaling, grand_normalization)) for subj in subjs]
for pp in processes:
	pp.start()
for pp in processes:
	pp.join()
		

# PRE,POST pcode, code within condition decoding
top_n = 350
for prepost in ["PRE","POST"]:
	for cond in ["text"]+["code"]*(prepost=="POST"):
		only_POST_subjs = ["NV_%.2d"%ii for ii in [4,12,13,15,17,18,25]]
		subjs = only_POST_subjs*(prepost=="POST")+["NV_%.2d"%ii for ii in [27,28,29,30,38,39,44,50,52,53,56,57,59,60,61,62,65,77]]
		label = "pyMVPA_pool_%s/%s[%dvtx]%s_%s"%(prepost,cond,top_n,"_grandSc"*grand_normalization,"%s")
		result_label = "pyMVPA_pool_%s/%s[%dvtx]%s"%(prepost,cond,top_n,"_grandSc"*grand_normalization)
		check_result(subjs=subjs, roi_list=roi_list, label=label, result_label=result_label)