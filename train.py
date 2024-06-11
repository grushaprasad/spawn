import argparse
import re
import math
import random
from copy import deepcopy
import numpy as np
import pickle

from model import actr_model
import supertagger
import csv
import os


train_fname = './data/train10000.txt'

# def load_chunks(model_name):
# 	for chunk_type in ['lexical', syntax]
# 	fname = f'./declmem/{chunk_type}_chunks_{model_name}.pkl'
# 	with open()


with open('./declmem/lexical_chunks_ep.pkl', 'rb') as f:
	lexical_chunks_ep = pickle.load(f)

with open('./declmem/syntax_chunks_ep.pkl', 'rb') as f:
	syntax_chunks_ep = pickle.load(f)

with open('./declmem/syntax_chunks_wd.pkl', 'rb') as f:
	syntax_chunks_wd = pickle.load(f)

with open('./declmem/lexical_chunks_wd.pkl', 'rb') as f:
	lexical_chunks_wd = pickle.load(f)

with open('./declmem/syntax_chunks_wd2.pkl', 'rb') as f:
	syntax_chunks_wd2 = pickle.load(f)

with open('./declmem/lexical_chunks_wd2.pkl', 'rb') as f:
	lexical_chunks_wd2 = pickle.load(f)

with open('./declmem/type_raising_rules.pkl', 'rb') as f:
	type_raising_rules = pickle.load(f)

with open('./declmem/null_mapping.pkl', 'rb') as f:
		null_mapping = pickle.load(f)

with open('./declmem/null_mapping_wd2.pkl', 'rb') as f:
		null_mapping_wd2 = pickle.load(f)

with open(train_fname,'r') as f:
	train_dat = f.readlines()


train_dat = [x.strip() for x in train_dat]


# Hyperparameters that do not change
decay = 0.5           # Standard value for ACT-R models
max_activation = 1.5  # Standard value for ACT-R models
latency_exponent = 1  # Will always set this to 1 (because then it matches eqn 4 from Lewis and Vasisth)


models = ['WD', 'WD2', 'EP']


parser = argparse.ArgumentParser(description='SPAWN Trainer')


## Directory parameter
parser.add_argument('--traindir', type=str, default='../trained_models/',
                help='path to save models')

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

if not os.path.isdir(args.traindir):
	os.makedirs(args.traindir)

if args.reanalysis == 'uncertainty10':
	temperature = 10
else:
	temperature = 1 # temp doesn't matter for start reanalysis

def append_to_log(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')


# num_sents = int(input('Num training sents: ').strip())
# global_sd = input('Global SD: ').strip()
# reanalysis_type = input('Reanalysis type (start, uncertainty): ').strip()

failed_sents = []
sds = []
for i in range(args.num_parts):
	fname = f'train{args.num_train/1000}_{args.reanalysis_type}_sd{args.global_sd_dist}-{args.global_sd_param1}-{args.global_sd_param2}_giveup{args.giveup}_m{i}.pkl'

	#set seed
	seed = i
	random.seed(seed)
	np.random.seed(seed)

	#sample model instance hyperparameters
	latency_factor = np.random.beta(2,6)  #prior  Englemann and Vasisth chapter. 

	if args.global_sd_dist == 'normal':
		noise_sd = abs(np.random.normal(args.global_sd_param1, args.global_sd_param2)) #param1=mean, param2 = sd
	else: #uniform
		noise_sd = np.random.uniform(args.global_sd_param1, args.global_sd_param2) #param1=lower, param2 = upper

	sds.append([i, noise_sd])

	# Get training data
	curr_train_dat = deepcopy(train_dat)
	random.shuffle(curr_train_dat)
	curr_train_dat = curr_train_dat[:args.num_train]

	
	## Train EP
	if 'EP' in models:
		random.seed(seed) #reset seeds right before training to be as close to WD as possible
		np.random.seed(seed)
		actr_model_ep = actr_model(decay,
								  max_activation,
								  noise_sd,
								  latency_factor,
							  	  latency_exponent,
								  syntax_chunks_ep,
								  lexical_chunks_ep,
								  type_raising_rules,
								  null_mapping,
								  args.giveup,
								  args.reanalysis,
								  temperature)

		# print('Training EP')
		for ind,sent in enumerate(curr_train_dat):
			# print(sent)
			# if ind%100 == 0:
			# 	print(f'Trained {ind} sentences')
			actr_model_ep.update_counts([sent])
			actr_model_ep.update_base_activation()
			actr_model_ep.update_lexical_activation()

		failed_sents.append(['EP', i, noise_sd, actr_model_ep.num_failed_sents])

		ep_fname = f'{args.traindir}/ep_{fname}'

		with open(ep_fname, 'wb') as f:
			pickle.dump(actr_model_ep, f)

	if 'WD' in models:
		random.seed(seed)  #reset seeds to be as close to EP as possible
		np.random.seed(seed)

		actr_model_wd = actr_model(decay,
							  max_activation,
							  noise_sd,
							  latency_factor,
							  latency_exponent,
							  syntax_chunks_wd,
							  lexical_chunks_wd,
							  type_raising_rules,
							  null_mapping,
							  args.giveup,
							  args.reanalysis,
							  temperature)

		for ind,sent in enumerate(curr_train_dat):
			# print(sent)
			# if ind%100 == 0:
			# 	print(f'Trained {ind} sentences')
			actr_model_wd.update_counts([sent])
			actr_model_wd.update_base_activation()
			actr_model_wd.update_lexical_activation()

		failed_sents.append(['WD', i, noise_sd, actr_model_wd.num_failed_sents])

		wd_fname = f'{args.traindir}/wd_{fname}'

		with open(wd_fname, 'wb') as f:
			pickle.dump(actr_model_wd, f)


	## Train WD2
	if 'WD2' in models:
		random.seed(seed)  #reset seeds to be as close to EP as possible
		np.random.seed(seed)

		actr_model_wd2 = actr_model(decay,
							  max_activation,
							  noise_sd,
							  latency_factor,
							  latency_exponent,
							  syntax_chunks_wd2,
							  lexical_chunks_wd2,
							  type_raising_rules,
							  null_mapping_wd2,
							  args.giveup,
							  args.reanalysis,
							  temperature)

		for ind,sent in enumerate(curr_train_dat):
			# print(sent)
			# if ind%100 == 0:
			# 	print(f'Trained {ind} sentences')
			actr_model_wd2.update_counts([sent])
			actr_model_wd2.update_base_activation()
			actr_model_wd2.update_lexical_activation()

		failed_sents.append(['WD2', i, noise_sd, actr_model_wd2.num_failed_sents])

		wd2_fname = f'{args.traindir}/wd2_{fname}'

		with open(wd2_fname, 'wb') as f:
			pickle.dump(actr_model_wd2, f)

	if i%100 == 0:
		text = f'Processed {i+1} participants'
		append_to_log(text, args.progress_fname)


failed_sent_fname = f'../trained_models/failed_sents_train{args.num_train/1000}_{args.reanalysis_type}_sd{args.global_sd_dist}-{args.global_sd_param1}-{args.global_sd_param2}_giveup{args.giveup}.csv'

# with open(sd_fname, 'w') as f:
with open(failed_sent_fname, 'w') as f:
	writer = csv.writer(f, delimiter = ',')

	writer.writerow(['model', 'seed', 'noise_sd', 'num_failed'])

	for item in failed_sents:
		writer.writerow(item)

