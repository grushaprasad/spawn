import re
import argparse
import math
import random
from copy import deepcopy
import numpy as np
import pickle

import supertagger
# import train
import csv
import os

from model import actr_model
import pandas as pd



def generate_priming_preds(model_fname, stim_fname, part_id):
	preds = []
	with open(stim_fname, 'r') as sf:
		stims = sf.readlines()

	stims = [x.lower() for x in stims]
	stims = [x.strip() for x in stims]

	with open(model_fname, 'rb') as mf:
		model = pickle.load(mf)

	for sent_id,sent in enumerate(stims):
		if len(sent.split()) > 3:
			model.update_counts([sent])
			model.update_base_activation()
			model.update_lexical_activation()
		else:
			partial_states = [{'left': 'DP', 'right': 'PP', 'combinator': '/'},
					          {'left': 'TP', 'right': 'DP', 'combinator': '/'}]

			parse_states, tags, words, act_vals = supertagger.supertag_sentence(model, sent, end_states=partial_states)
			num_retries = model.num_retries
			
			if tags == None:
				passive = 'NA'
				final_state = 'NA'
				final_state_rule = 'NA'
			else:
				final_state = parse_states[-1]
				final_state_rule = final_state['left'] + final_state['combinator'] + final_state['right']
				if tags[-1] == 'Vt_pass':
					passive = 1
				else:
					passive = 0

			preds.append([stim_fname,part_id,sent_id,passive, final_state_rule, num_retries])

	return(preds)

def flatten(l):
	return([x for sublist in l for x in sublist])



def save_preds(preds, fname):
	with open(fname, 'w') as f:
		writer = csv.writer(f, delimiter = ',')
		writer.writerow(['list', 'part_id', 'sent_id', 'passive', 'goalstate', 'num_retries'])

		for pred in preds:
			writer.writerow(pred)



parser = argparse.ArgumentParser(description='SPAWN Prime')

## Directory
parser.add_argument('--traindir', type=str, default='../trained_models/',
                help='path where models are saved')

parser.add_argument('--preddir', type=str, default='../predictions/',
                help='path to save predictions')

parser.add_argument('--progress_fname', type=str, 
                help='fname+path to save progress')

## Model hyperparameters
parser.add_argument('--reanalysis', type=str, default='uncertainty',
                choices=['start', 'uncertainty1', 'uncertainty10'],
                help='type of reanalysis mechanism')

parser.add_argument('--global_sd_dist', type=str, default='normal',
                choices=['uniform', 'normal'],
                help='Distribution for global_sd')

parser.add_argument('--global_sd_param1', type=float, default=0.35,
                help='mean for normal, lower range for uniform')

parser.add_argument('--global_sd_param2', type=float, default=1.0,
                help='sd for normal, upper range for uniform')

parser.add_argument('--giveup', type=int, default=1000,
                help='number of iterations before abandoning prior')

parser.add_argument('--num_train', type=int, default=0,
                help='number of sentences to train on; max 10000')


## Experiment hyperparameters

parser.add_argument('--seed', type=int, default=7,
                help='random seed')

parser.add_argument('--num_parts', type=int, default=1280,
                help='number of iterations before abandoning prior')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

if not os.path.isdir(args.preddir):
	os.makedirs(args.preddir)

def append_to_log(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')

with open(args.progress_fname, 'w') as file:
    file.write('Starting\n')

stim_fnames = []


for a in ['1','2','3','4']:
	for b in ['A', 'B', 'C', 'D']:
		for r in ['', '_rev']:
			stim_fnames.append(f'./data/stimuli/list_{a}{b}{r}.txt')


preds = {'ep':[], 'wd':[], 'wd2':[]}

for i in range(args.num_parts):
	curr_stim_fname = stim_fnames[i%32]

	for model in preds:

		random.seed(i)
		np.random.seed(i)

		fname = f'{args.traindir}{model}_train{args.num_train/1000}_{args.reanalysis}_sd{args.global_sd_dist}-{args.global_sd_param1}-{args.global_sd_param2}_giveup{args.giveup}_m{i}.pkl'

		curr_preds = generate_priming_preds(fname, curr_stim_fname, i)

		preds[model].extend(curr_preds)


	if i%10 == 0:
		text = f'Processed {i+1} participants'
		append_to_log(text, args.progress_fname)

for model in preds:
	fname = f'{args.preddir}{model}_train{args.num_train/1000}_{args.reanalysis}_sd{args.global_sd_dist}-{args.global_sd_param1}-{args.global_sd_param2}_giveup{args.giveup}.csv'

	save_preds(preds[model], fname)






		


