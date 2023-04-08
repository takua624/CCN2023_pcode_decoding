
# -*- coding: utf-8 -*-

from psychopy import core, visual, event, gui
from psychopy.hardware import cedrus
import pyxid2 as pyxid
import pandas as pd
import numpy as np
import os
import time
import random
import sys
devices = pyxid.get_xid_devices()
if devices: 
    rb = devices[0]
else:
    print("No button box, use keyboards!")
print(devices)

############################
skip_runs = [] 
# change this if the scan doesn't start from the first run
# e.g., if the batches after shuffle are [4,1,3,2]
# when we skip run 0, the remaining batches to run are [1,3,2]

dlg = gui.Dlg(title="Info")
dlg.addField("Subj.ID", "myself")
dlg.addField("Batches", "1,2,3,4,5,6,7,8,9") 
# You can copy and paste from here:
# PRE, Pcode only: 1,2,3,4,5,6,7,8,9
# practice for PRE: 98
# POST, Pcode+code: 10,11,12,13,14,15,16,17,18
# practice for POST: 99 (the 2 text pcode are moved to batch 999, not using here. Because by now the participants should have done enough trials of pcode)
# behavioral pcode reading: -1 (if testing for the second time, use -2)
# practice for behavioral: 0
dlg.addField("Batch rand seed", "0")
dlg.addField("Item rand seed", choices=[1,2,3,4,5,6])
# Item rand seed also controls whether someone sees the same batch of pseudocode in the PRE session or the POST session
# 2,4,6 --> PRE; 1,3,5 --> POST
dlg.addField("Answer group", choices=[1,2])
dlg.addField("Session", choices=["PRE","POST"])
dlg.addField("Scr.Width", "1600")
dlg.addField("Scr.Height","1200")
dlg.addField("Behavioral?", initial=False)
dlg.addField("Test function", initial=False)


info = dlg.show()
if dlg.OK:
	print(info)

test_function = info[-1]
behavioral = info[-2]
session = info[5]
# ===============================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MACHINE DEPENDENT PARAMETERS!!!
aspect = (int(info[6]),int(info[7]))
print(aspect)
# (960, 540) on my laptop (2x disp size)
# (1920, 1080) on the working laptop
# (1600,1200) in scanner

pic_size = 2
# 2 on my laptop (2x disp size)
# 2 on the working laptop

pic_size_mod = (0,0.25)
# (0,0.25) on my laptop (2x disp size)
# (0,0.25) on the working laptop

position = (0,0) # should be (-0.5, 0.5 on retina mac)
txt_h=0.05
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ===============================

# EXTRACT SUBJ AND RUN INFO FROM DIALOGUE BOX #
sj_ID = info[0]

batch_seed = int(info[2])
item_seed = int(info[3])
run_vec = info[1]
ans_group = int(info[4])
run_ID_all = [int(dd) for dd in run_vec.split(",")]

random.seed(batch_seed)
if not behavioral:
    random.shuffle(run_ID_all)
print(run_ID_all)
# all the participants see the same batches in one session
# but each participants sees the batches in a different order

#================================

# SET UP THE TIMINGS FOR THE EVENTS
init_cross_time = 0.5
present_time = 20 - 19*test_function # should be 24 without the input phase
# 210913: let's try 20s
input_time = 6 - 5*test_function
# 24s comes from the previous experiment
mid_cross_time = 0.5
question_time = 4 - 3*test_function
t_ITI = 5 - 4.5*test_function # should be 5
if behavioral:
    t_ITI = 2*t_ITI
time_per_trial = (init_cross_time + present_time + input_time + mid_cross_time + question_time + t_ITI)
feedback_time = 5

cont_key = "space"
true_key = 0
false_key = 1
skip_key = 6 # should be 6
scan_skip = 2
true_key_kb = '1'
false_key_kb = '0'
present_time_refractory = 2 - 2*test_function
question_time_refractory = 0.2 - 0.2*test_function
trigger = [5] # the keys in the scanner are labeled 1-6, but recognized by Python as 0-5!!

#================================

# Create a window to draw in
# !!!!!!!!!!! MUST TEST WITH THE TEST SCREEN TO DECIDE THE ASPECT RATIO!!!!!!!
win = visual.Window(size=aspect, screen=1, allowGUI=False, color='black', monitor="testMonitor", winType='pyglet')
for ww in win.size:
	ww = ww/2

#================================
stimuli_roster = pd.read_excel("item_roster.xlsx")

# determine the order of items in a run/batch
def shuf_item_order(rii, s_r):
    
    # the random seed is different for every run
    random.seed(item_seed+rii)
    
    # Find out the most frequent condition
    # Suppose condition A appears 5 times, 
    # and condition B and C appear 3 times each.
    # Suppose the row ids of A are [0,1,2,3,4]
    # B is [5,6,7], # C is [8,9,10].
    # Expand the row ids of B and C into [5,6,7,-1,-1] and [8,9,10,-1,-1].
    # Shuffle the three row ids.
    # Stack them to make a 3-by-5 array, such as:
    # [[ 4, 1, 3, 2, 0],
    #  [-1, 5,-1, 7, 6],
    #  [ 9,10,-1,-1, 8]]
    # For each column(=repetition), shuffle the conditions.
    # Concatenate the columns to derive the sequence of presentation.
    # Then remove the -1 entries, DONE!
    mode_item = s_r["cond"].mode()[0]
    repetition = s_r[s_r["cond"]==mode_item].shape[0]
    conds = s_r["cond"].unique()
    arr = []
    yes_no = {cc:[] for cc in conds}
    less_item = (rii%2==1)
    for cc in conds:
        ids = list(s_r[s_r["cond"]==cc].index)
        yes_no[cc] = [int(less_item)]*(len(ids)//2) + [int(not less_item)]*(len(ids)-len(ids)//2)
        ids += [-1]*(repetition-len(ids))
        ids = random.sample(ids, len(ids))
        arr += [ids]
        less_item = not less_item
    arr = np.array(arr)
#    print(arr)
    
    order = []
    for rr in range(repetition):
        ids = list(arr[:,rr])
        ids = random.sample(ids, len(ids))
        order += ids
    order = [ii for ii in order if ii!=-1]
    s_r = s_r.iloc[order]
#    print(list(s_r["cond"]))
    
    # Determine the distribution of YES's and NO's
    half = len(order)//2
    yes_no = list(yes_no.values())
    yes_no = [x for y in yes_no for x in y]
    if ans_group == 2:
        yes_no = [1-x for x in yes_no]
    yes_no = [yes_no[x] for x in order]
#    yes_no = [1]*half + [0]*(len(order)-half)
#    random.shuffle(yes_no)
    s_r = s_r.copy()
    s_r.loc[:,"yes_no"] = yes_no
    return s_r
    
# determine the order of items in a run/batch
# using the best sequences to eliminate carry over effect
# so the number of stimuli for each condition is the same
def shuf_best_sequence(rii, s_r):
    conds = list(s_r["cond"].unique())
#    print(conds)
    n_conds = len(conds)
    conds = conds[batch_seed%n_conds:]+conds[:batch_seed%n_conds]
    
    # the random seed is different for every run
    random.seed(item_seed+rii)
    
    # the best sequence to eliminate carry over effect:
    seq_df = pd.read_excel("item_order.xlsx", sheet_name="%dconds"%n_conds)
    the_seq = list(seq_df["run%d"%(rii+1)])
#    print(the_seq)
    
    arr = []
    yes_no_arr = []
    n_rep = max([s_r[s_r["cond"]==cc].shape[0] for cc in conds])
    repetition = {cc:0 for cc in conds}
#    print(n_rep)
    
    for cc in conds:
        ids = list(s_r[s_r["cond"]==cc].index)
        ids += [-1]*(n_rep-len(ids))
        yes_no = [int(ans_group==1)]*(len(ids)//2)+[int(ans_group==2)]*(len(ids)-len(ids)//2)
        ids = random.sample(ids, len(ids))
        yes_no = random.sample(yes_no, len(yes_no))
        arr += [ids]
        yes_no_arr += [yes_no]
    arr = np.array(arr)
    yes_no_arr = np.array(yes_no_arr)
#    print(arr)
#    print(yes_no_arr)

    order = []
    yes_no_order = []
    for ss in the_seq:
#        print(repetition[conds[ss-1]])
        order += [arr[ss-1, repetition[conds[ss-1]]]]
        yes_no_order += [yes_no_arr[ss-1, repetition[conds[ss-1]]]]
        repetition[conds[ss-1]] += 1
    yes_no_order = [yes_no_order[ii] for ii in range(len(yes_no_order)) if order[ii]!=-1]
    order = [ii for ii in order if ii!=-1]
#    print(order)
#    orig_order = np.array(range(len(order)))
#    yes_no_order = list(yes_no_order[orig_order[order]])
    s_r = s_r.iloc[order]
    s_r = s_r.assign(yes_no=yes_no_order)
#    
#    
    return s_r
    
def countdown_rect(t_now, t_limit, win, answered=False):
    rect_color = "#808080" if not answered else [0,0.7,0]
    percentage = 1.0-(t_now/t_limit)
    tokens = visual.Rect(win=win, height=0.01, width=2*percentage, fillColor=rect_color)
    tokens.pos = [-1+percentage,-0.85]
    tokens.draw()
    return


#check_items = pd.DataFrame()

for rii in range(len(run_ID_all)):
    if rii in skip_runs:
        continue
    run_ID = run_ID_all[rii] # run_ID = batch number
    s_r = stimuli_roster[stimuli_roster["batch%d"%(item_seed%2+1)]==run_ID].reset_index().drop(columns=["index","used","content"])
    
#    print(s_r.columns)
    if 19>run_ID>0:
        s_r = shuf_best_sequence(rii, s_r)
    else:
        s_r = shuf_item_order(rii,s_r)
#    print("###################"+str(s_r.shape))
#    aa = list(s_r["cond"])
#    print()
#    check_items["Run%d=Batch%d"%(rii+1,run_ID)] = pd.Series(aa)
#    print(s_r)
    
    # Setting up the subject response output format
    out_file = "subj_response/%s_%s_run%d(batch%d)_g%d_itemseed%d.xlsx"%(sj_ID,session,rii+1,run_ID, ans_group, item_seed)
#    s_r.to_excel(out_file, index=False)

    o_f = pd.DataFrame(columns=list(s_r.columns)+
          ["onset", "offset", "read_onset", "read_offset", "read_time", "input_onset", "input_offset", "compute_time",
           "q_onset", "q_offset", "q_time", "sj_ans", "correct"])
    fixation_cross = visual.TextStim(win, pos=position, text="+")
    wait_text = visual.TextStim(win, pos=position, text="The experiment will begin soon.",
                color="white", anchorHoriz="center", anchorVert="center", height=txt_h*1.6)
                
    # Get ready for the trigger
    event.clearEvents()
    if devices:
        rb.clear_response_queue()
        rb.reset_base_timer()
        rb.reset_rt_timer()
        rb.con.flush()
    print("Waiting for the trigger.")
    ced_no_response = True
    while ced_no_response and (not event.getKeys(["escape"])):
        if devices:
            ced_no_response = ((not rb.response_queue) or (not rb.response_queue[-1]['key'] in trigger))
            rb.poll_for_response()
        wait_text.draw()
        win.flip()
    t_start = time.time()
    print("Welcome, Trigger-san!")
    
    # Print out the basic info of this run
    print()
    print("...NOW RUNNING: RUN %d (BATCH %d)"%(rii+1, run_ID))
    print("...SUBJECT: %s"%sj_ID)
    print("... ITEM SEED: %d"%item_seed)
    for cc in s_r["cond"].unique():
        n_items = s_r[s_r["cond"]==cc].shape[0]
        print("... COND [%s]: %d ITEMS"%(cc,n_items))
    print()
    
    trial_count = 0
    curr_trial_num = 1
    for ii,row in s_r.iterrows():
        stim_file = row["stem"]+"_pic%d.png"%row["pic_no"]
        input_file = row["stem"]+"_pic%d[input].png"%row["pic_no"]
        ques_file = row["stem"]+"_pic2(%s).png"%("yes"*(row["yes_no"]==1)+"no"*(row["yes_no"]==0))
        cross_file = row["stem"]+"_fix.png"
        correct_answer = row["yes_no"]
        o_f = o_f.append(row)
        
        second_cross = visual.ImageStim(win, image=cross_file, units="norm", size=aspect, pos=position)
        second_cross.size = pic_size
        second_cross.size += pic_size_mod
        second_cross.pos=position
        
        # Default: subject didn't answer
        sj_ans = -1
        correct = 0
        
        # Initial fixation cross
        trial_clock = core.Clock()
        trial_onset = time.time()-t_start
        o_f.at[ii,"onset"] = trial_onset
        t = 0
        while (not event.getKeys(["escape"])) and (t<=init_cross_time):
            t = trial_clock.getTime()
            fixation_cross.draw()
            win.flip()
        
        # Present the stimuli
        event.clearEvents()
        trial_clock = core.Clock()
        t = 0
        present_pic = visual.ImageStim(win, image=stim_file, units="norm", size=aspect, pos=position)
        present_pic.size = pic_size
        present_pic.size += pic_size_mod
        present_pic.pos=position
        t_onset = time.time()-t_start
        o_f.at[ii,"read_onset"] = t_onset
        if devices:
            rb.clear_response_queue()
            rb.con.flush()
            rb.reset_rt_timer()
        ced_no_response = True
        while ced_no_response and (not event.getKeys([cont_key.lower(),"escape"])) and (t<=present_time):
            if devices and behavioral:
                rb.poll_for_response()
                ced_no_response = ((not rb.response_queue) or (not rb.response_queue[-1]['key'] in [skip_key]) or (not rb.response_queue[-1]['pressed']))
                if not ced_no_response:
                    rt = rb.response_queue[-1]['time']
#            if t<present_time_refractory:
#                rb.clear_response_queue()
#                rb.con.flush()
            present_pic.draw()
            t = trial_clock.getTime()
            t_cdown = 1*(t//1) # shorten the countdown bar every 2 seconds
            countdown_rect(t_cdown,present_time,win)
            win.flip()
        read_time = t if ced_no_response else rt/1000
        t_offset = time.time()-t_start
        o_f.at[ii,"read_offset"] = t_offset
        o_f.at[ii,"read_time"] = read_time
        
        # Present the input
        event.clearEvents()
        trial_clock = core.Clock()
        t = 0
        present_pic = visual.ImageStim(win, image=input_file, units="norm", size=aspect, pos=position)
        
        present_pic.size = pic_size
        present_pic.size += pic_size_mod
        present_pic.pos=position
        t_onset = time.time()-t_start
        o_f.at[ii,"input_onset"] = t_onset
        if devices:
            rb.clear_response_queue()
            rb.con.flush()
            rb.reset_rt_timer()
        ced_no_response = True
        # 211114: this part should always be self-paced to save time
        while ced_no_response and (not event.getKeys([cont_key.lower(),"escape"])) and (t<=input_time):
            if devices:
                rb.poll_for_response()
                ced_no_response = ((not rb.response_queue) or (not rb.response_queue[-1]['key'] in [skip_key, scan_skip]) or (not rb.response_queue[-1]['pressed']))
                if not ced_no_response:
                    rt = rb.response_queue[-1]['time']
#            if t<present_time_refractory:
#                rb.clear_response_queue()
#                rb.con.flush()
            present_pic.draw()
            t = trial_clock.getTime()
            t_cdown = t//1 # shorten the countdown bar every second
            countdown_rect(t_cdown,input_time,win)
            win.flip()
        compute_time = t if ced_no_response else rt/1000
        t_offset = time.time()-t_start
        o_f.at[ii,"input_offset"] = t_offset
        o_f.at[ii,"compute_time"] = compute_time
        
        # intermediate fixation cross
        trial_clock = core.Clock()
        t = 0
        while (not event.getKeys(["escape"])) and (t<=mid_cross_time):
            t = trial_clock.getTime()
#            fixation_cross.draw()
            second_cross.draw()
            win.flip()
         
        # Present the question
        event.clearEvents()
        trial_clock = core.Clock()
        t = 0
        question_pic = visual.ImageStim(win, image=ques_file, units="norm", size=aspect, pos=position)
        question_pic.size = pic_size
        question_pic.size += pic_size_mod
        question_pic.pos=position
        t_onset = time.time()-t_start
        o_f.at[ii,"q_onset"] = t_onset
        if devices:
            rb.clear_response_queue()
            rb.con.flush()
            rb.reset_rt_timer()
        ced_no_response = True
        answered = False
        go_on = True
        while go_on:
            if devices:
                rb.poll_for_response()
#            if t<question_time_refractory:
#                rb.clear_response_queue()
#                rb.con.flush()
                if rb.response_queue and not answered:
                    response = rb.response_queue[-1]['key']
                    sj_ans = int(response==true_key)
                    correct = int(sj_ans==correct_answer)
                    if response in [true_key, false_key]:
                        answered = True
                        ans_time = rb.response_queue[-1]['time']/1000
            if not devices and not answered:
                kb_response = event.getKeys([true_key_kb, false_key_kb])
                if kb_response and not answered:
                    sj_ans = int(kb_response[0]==true_key_kb)
                    correct = int(sj_ans==correct_answer)
                    answered = True
                    ans_time = trial_clock.getTime()
            question_pic.draw()
            t = trial_clock.getTime()
            t_cdown = t
            countdown_rect(t_cdown,question_time,win, answered)
            
            # 211114: we decided that this part is always self-paced
            go_on = (not event.getKeys(["escape"])) and (t<=question_time) and not answered
#            if behavioral:
#                go_on = (not event.getKeys(["escape"])) and (t<=question_time) and not answered
#            else:
#                go_on = (not event.getKeys(["escape"])) and (t<=question_time)
            win.flip()
        if not answered:
            ans_time = trial_clock.getTime()
        q_time = t
        t_offset = time.time()-t_start
        o_f.at[ii,"q_offset"] = t_offset
        o_f.at[ii,"offset"] = t_offset
        o_f.at[ii,"q_time"] = ans_time
        o_f.at[ii,"sj_ans"] = sj_ans
        o_f.at[ii,"correct"] = correct
        print("Trial %d"%curr_trial_num)
        print("COND: %s"%row["cond"])
        print("READ_T: %f; RESP_T: %f, %s"%(read_time, q_time, "RIGHT"*correct+"WRONG"*(not correct)))
        print()
        curr_trial_num += 1
        
        # ITI
        event.clearEvents()
        trial_clock = core.Clock()
        t = 0
        if behavioral:
            show_ITI = visual.TextStim(win, pos=(0,0),text="The next trial will begin soon. You may press SPACE(or the RED button) to continue immediately.", color="#777777", anchorHoriz='center', anchorVert='center', height=txt_h*1.5, alignText='left')
        if not behavioral:
            show_ITI = visual.TextStim(win, pos=(0,0),text="", color="#777777", anchorHoriz='center', anchorVert='center', height=txt_h*1.5, alignText='left')

        if devices:
            rb.clear_response_queue()
            rb.con.flush()
            rb.reset_rt_timer()
        ced_no_response = True
        while (ced_no_response) and (not event.getKeys([true_key_kb, false_key_kb, "space", "escape"])) and (t<=t_ITI): # use if self-paced
            if devices and behavioral: # use if self-paced
                rb.poll_for_response() # use if self-paced
                ced_no_response = ((not rb.response_queue) or not rb.response_queue[-1]['key'] in [true_key, false_key, skip_key] or not rb.response_queue[-1]['pressed']) # use if self-paced
            show_ITI.draw()
            t = trial_clock.getTime()
            if behavioral:
                countdown_rect(t,t_ITI,win)
            win.flip()
      
        o_f.to_excel(out_file, index=False)
  
    NN = float(o_f.shape[0])
    acc_ratio = float(o_f["correct"].sum())/NN
    acc = 100*acc_ratio
    speed_raw = o_f["q_time"].mean()
    print("Performance in this run: ACC = %f, RT = %f"%(acc_ratio,speed_raw))
    print("[PLEASE STOP THE CURRENT SCAN]")
    speed_scale = 1-((speed_raw-1)/(question_time-1))
    speed_scale = 1 if speed_scale>1 else speed_scale
    speed_meter = [speed_scale>0.8, speed_scale>0.6, speed_scale>0.4, speed_scale>0.2, speed_scale>=0]
    speed_text_dict = {1:"Very\nSlow", 2:"Slow", 3:"Medium", 4:"Fast", 5:"Very\nFast"}
    speed_text = speed_text_dict[sum(speed_meter)]


    feedback_title = visual.TextStim(win, pos=(0,0.7), text="Here's your performance in this run!", color="white", anchorHoriz="center", anchorVert="center", height=txt_h*1.6)
    feedback_acc = visual.TextStim(win, pos=(-0.2,-0.6), text="Accuracy", color="white", anchorHoriz="center", anchorVert="center", height=txt_h*1.8)
    feedback_acc_val = visual.TextStim(win, pos=(-0.1,0), text="%.0f %%"%(acc), color="white", anchorHoriz="right", anchorVert="center", height=txt_h*2.1)
    feedback_speed = visual.TextStim(win, pos=(0.2,-0.6), text="Speed", color="white", anchorHoriz="center", anchorVert="center", height=txt_h*1.8)
    feedback_speed_text = visual.TextStim(win, pos=(0.1,0), text=speed_text, color="white", anchorHoriz="left", anchorVert="center", height=txt_h*2.1)

    all_bar_width = 0.1
    acc_frame = visual.Rect(win=win, height=1, width=all_bar_width, fillColor="black")
    acc_frame.pos = [-0.2, 0]
    speed_frame = visual.Rect(win=win, height=1, width=all_bar_width, fillColor="black")
    speed_frame.pos = [0.2,0]

    acc_bar = visual.Rect(win=win, height=acc_ratio, width=all_bar_width, fillColor=[0.3,0.9,0.3], lineWidth=0)
    acc_bar.pos = [-0.2, -(1-acc_ratio)/2]
    speed_bar = visual.Rect(win=win, height=speed_scale, width=all_bar_width, fillColor=[0.3,0.9,0.3], lineWidth=0)
    speed_bar.pos = [0.2, -(1-speed_scale)/2]

    event.clearEvents()
    trial_clock = core.Clock()
    t = 0
    while (not event.getKeys(["escape"])) and (t<=feedback_time):
        t = trial_clock.getTime()
        feedback_title.draw()
        feedback_acc.draw()
        feedback_acc_val.draw()
        feedback_speed.draw()
        feedback_speed_text.draw()
        acc_frame.draw()
        speed_frame.draw()
        acc_bar.draw()
        speed_bar.draw()
        win.flip()

#check_items.to_excel("subj_response/check_item.xlsx",index=False)
core.quit()
print("done")
