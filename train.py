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


random.seed(7)
np.random.seed(7)


# num_sents = 2000000
# wd_fname = './trained_models/wd_trained_2M.pkl' 
# ep_fname = './trained_models/ep_trained_2M.pkl' 

# num_sents = 10000
# wd_fname = './trained_models/wd_trained_10k_3amv.pkl' 
# ep_fname = './trained_models/ep_trained_10k_3amv.pkl' 


# wd_fname = './trained_models/wd_trained_1k_3amv.pkl' 
# ep_fname = './trained_models/ep_trained_1k_3amv.pkl'
train_fname = './data/train10000.txt'
# train_fname = './test_sents_supertagger.txt'



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

if not os.path.isdir('./trained_models/'):
	os.makedirs('./trained_models/')


train_dat = [x.strip() for x in train_dat]


# num_sents = 5000
# num_parts = 512
# num_parts = 1
num_parts = 1280
# global_sd = 'uniform'
# global_sd = 1

# Hyperparameters that do not change
decay = 0.5           # Standard value for ACT-R models
max_activation = 1.5  # Standard value for ACT-R models
latency_exponent = 1  # Will always set this to 1 (because then it matches eqn 4 from Lewis and Vasisth)

# max_iters = 1000
max_iters = 2000  # Number of iterations before "giving up" and starting afresh. After half of these iterations parsing ignores priors

models = ['WD', 'WD2', 'EP']
# models = ['WD2']

#for num_sents in [100,1000,10000]:
#for num_sents in [100,1000]:
if __name__ == '__main__':
	num_sents = int(input('Num training sents: ').strip())
	global_sd = input('Global SD: ').strip()
	reanalysis_type = input('Reanalysis type (start, uncertainty): ').strip()

	try:
		global_sd = float(global_sd)
	except:
		print('Non numeric sd entered')
		global_sd = global_sd

	print('Num sents: ', num_sents)
	failed_sents = []
	sds = []
	for i in range(num_parts):
		seed = i
		if i%200 == 0:
			print(f'Processed {i} participants')
		random.seed(seed)   #different seeds used for fitting the error terms than generating participants
		np.random.seed(seed)

		latency_factor = np.random.beta(2,6)  #prior  Englemann and Vasisth chapter. 
		
		if type(global_sd) == int or type(global_sd) == float:
			noise_sd = abs(np.random.normal(0.35, global_sd))
		elif global_sd == 'uniform':
			noise_sd = np.random.uniform(0.2,0.5)
		else:
			sd_type = np.random.choice(['low_sd', 'high_sd'], p=[0.75,0.25])
			if sd_type == 'low_sd':
				noise_sd = abs(np.random.normal(0.35, 1))
			else:
				if global_sd == 'bimodal2.35':
					noise_sd = abs(np.random.normal(2.35, 1))
				else:
					noise_sd = abs(np.random.normal(3.35, 1))


		sds.append([i, noise_sd])

		curr_train_dat = deepcopy(train_dat)
		random.shuffle(curr_train_dat)

		curr_train_dat = curr_train_dat[:num_sents]

		

		

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
									  max_iters,
									  reanalysis_type)

			# print('Training EP')
			for ind,sent in enumerate(curr_train_dat):
				# print(sent)
				# if ind%100 == 0:
				# 	print(f'Trained {ind} sentences')
				actr_model_ep.update_counts([sent])
				actr_model_ep.update_base_activation()
				actr_model_ep.update_lexical_activation()

			failed_sents.append(['EP', i, noise_sd, actr_model_ep.num_failed_sents])


			ep_fname = './trained_models/%s_reanalysis/ep_train%sk_sd%s_part%s.pkl'%(reanalysis_type, str(num_sents/1000), str(global_sd), str(i))

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
								  max_iters,
								  reanalysis_type)


			# # Train WD

			random.seed(seed)  #reset seeds to be as close to EP as possible
			np.random.seed(seed)
			# print('Training WD')
			for ind,sent in enumerate(curr_train_dat):
				# print(sent)
				# if ind%100 == 0:
				# 	print(f'Trained {ind} sentences')
				actr_model_wd.update_counts([sent])
				actr_model_wd.update_base_activation()
				actr_model_wd.update_lexical_activation()

			failed_sents.append(['WD', i, noise_sd, actr_model_wd.num_failed_sents])

			wd_fname = './trained_models/%s_reanalysis/wd_train%sk_sd%s_part%s.pkl'%(reanalysis_type,str(num_sents/1000), str(global_sd), str(i))

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
								  max_iters,
								  reanalysis_type)


			# # Train WD

			random.seed(seed)  #reset seeds to be as close to EP as possible
			np.random.seed(seed)
			# print('Training WD')
			for ind,sent in enumerate(curr_train_dat):
				# print(sent)
				# if ind%100 == 0:
				# 	print(f'Trained {ind} sentences')
				actr_model_wd2.update_counts([sent])
				actr_model_wd2.update_base_activation()
				actr_model_wd2.update_lexical_activation()

			failed_sents.append(['WD', i, noise_sd, actr_model_wd2.num_failed_sents])

			wd2_fname = './trained_models/%s_reanalysis/wd2_train%sk_sd%s_part%s.pkl'%(reanalysis_type,str(num_sents/1000), str(global_sd), str(i))

			with open(wd2_fname, 'wb') as f:
				pickle.dump(actr_model_wd2, f)

		

	# sd_fname = './trained_models/noisesds_train%sk_sd%s.csv'%(str(num_sents/1000), str(global_sd))
	failed_sent_fname = './trained_models/failed_sents_train%sk_sd%s.csv'%(str(num_sents/1000), str(global_sd))

	# with open(sd_fname, 'w') as f:
	with open(failed_sent_fname, 'w') as f:
		writer = csv.writer(f, delimiter = ',')

		writer.writerow(['model', 'seed', 'noise_sd', 'num_failed'])

		for item in failed_sents:
			writer.writerow(item)











	# training_sents_ep, training_tags_ep = create_training_data(struc_probs, struc_sents_ep, struc_tags_ep, num_sents)

	# training_sents_ep, training_tags_ep = shuffle_lists(training_sents_ep, training_tags_ep)


	# def create_training_data(probs, sents, tags, nsents):
	# 	all_sents = []
	# 	all_tags = []
	# 	for struc in probs:
	# 		n = round(probs[struc]*nsents)
	# 		curr_sents = [sents[struc]]*n
	# 		curr_tags = [tags[struc]]*n
	# 		all_sents.extend(curr_sents)
	# 		all_tags.extend(curr_tags)

	# 	return(all_sents, all_tags)


# def shuffle_lists(l1, l2):
# 	temp = list(zip(l1,l2))
# 	random.shuffle(temp)

# 	sl1, sl2 = zip(*temp)

# 	sl1, sl2 = list(sl1), list(sl2)

# 	return(sl1, sl2)



# ## Defining frequency counts based on Roland et al RC counts per million in text corpora (not spoken)


# src_prob = (14182 + 15024 + 18229)/3000000

# orc_prob = (2943 +  1976 + 1802)/3000000

# orrc_prob = (5455 + 4746 + 3385)/3000000

# frc_prob = (77993 + 882 + 375)/3000000

# rrc_prob = (26841 + 3302 + 3918)/3000000

# prog_prob = (13567 + 150 + 129)/3000000 #set this to be passive infinitive since there wasn't estimation for progrrc. 

# amv_prob = 1 - (src_prob + orc_prob + orrc_prob + frc_prob + rrc_prob + prog_prob)

# struc_probs = {
# 	'src': src_prob,
# 	'orc': orc_prob,
# 	'orrc': orrc_prob,
# 	'frc': frc_prob,
# 	'rrc': rrc_prob,
# 	'progrrc': prog_prob/2,
# 	'progrc': prog_prob/2,
# 	'amv1': amv_prob/3,
# 	'amv2': amv_prob/3,
# 	'amv3': amv_prob/3
# 	# 'amv1': amv_prob

# }

# struc_sents_wd = {
# 	'src': 'the defendant who examined the lawyer liked the cat .',
# 	'orc': 'the defendant who the lawyer examined liked the cat .',
# 	'orrc': 'the defendant comp_del the lawyer examined liked the cat .',
# 	'frc': 'the defendant who was examined by the lawyer liked the cat .',
# 	'rrc': 'the defendant comp_del aux_del examined by the lawyer liked the cat .',
# 	'progrrc': 'the defendant comp_del aux_del being examined by the lawyer liked the cat .',
# 	'progrc': 'the defendant who was being examined by the lawyer liked the cat .',
# 	'amv1': 'the defendant examined the lawyer .',
# 	'amv2': 'the defendant was unreliable .',
# 	'amv3': 'the defendant arrived at the palace .',
# 	'amv4': 'the defendant examined the lawyer and liked the cat .'
	

# }

# struc_tags_wd = {
# 	'src': 'Det NP_CP compsubj Vt_act Det NP Vt_act Det NP eos',
# 	'orc': 'Det NP_CP compobj Det NP Vt_act Vt_act Det NP eos',
# 	'orrc': 'Det NP_CP_null compobj_null Det NP Vt_act Vt_act Det NP eos',
# 	'frc': 'Det NP_CP compsubj aux_pass Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'rrc': 'Det NP_CP_null compsubj_null aux_pass Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'progrrc': 'Det NP_CP_null compsubj_null aux_pass aux_prog Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'progrc': 'Det NP_CP compsubj aux_prog ProgP Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'amv1': 'Det NP Vt_act Det NP eos',
# 	'amv2': 'Det NP aux Adj eos',
# 	'amv3': 'Det NP Vt_act Prep Det NP eos',
# 	'amv4': 'Det NP Vt_act Det NP conj Vt_act Det NP eos'
# }


# struc_sents_ep = {
# 	'src': 'the defendant who examined the lawyer liked the cat .',
# 	'orc': 'the defendant who the lawyer examined liked the cat .',
# 	'orrc': 'the defendant comp_del the lawyer examined liked the cat .',
# 	'frc': 'the defendant who was examined by the lawyer liked the cat .',
# 	'rrc': 'the defendant examined by the lawyer liked the cat .',
# 	'progrrc': 'the defendant being examined by the lawyer liked the cat .',
# 	'progrc': 'the defendant who was being examined by the lawyer liked the cat .',
# 	# 'amv': 'the defendant examined the lawyer and was unreliable'
# 	'amv1': 'the defendant examined the lawyer',
# 	'amv2': 'the defendant was unreliable .',
# 	'amv3': 'the defendant arrived at the palace .',
# 	'amv4': 'the defendant examined the lawyer and liked the cat .'
# }


# # struc_tags_ep = {
# # 	'src': 'Det NP compsubj Vt_act Det NP Vt_act Det NP',
# # 	'orc': 'Det NP compobj Det NP Vt_act Vt_act Det NP',
# # 	'orrc': 'Det NP compobj Det NP Vt_act Vt_act Det NP',
# # 	'frc': 'Det NP compsubj aux Vt_pass Prep Det NP Vt_act Det NP',
# # 	'rrc': 'Det NP Vt_pass Prep Det NP Vt_act Det NP',
# # 	'progrrc': 'Det NP aux_prog Vt_pass Prep Det NP Vt_act Det NP',
# # 	# 'amv': 'Det NP Vt_act Det NP conj aux Adj',
# # 	'amv': 'Det NP Vt_act Det NP'
# # }

# struc_tags_ep = {
# 	'src': 'Det NP_CP compsubj Vt_act Det NP Vt_act Det NP eos',
# 	'orc': 'Det NP_CP compobj Det NP Vt_act Vt_act Det NP eos',
# 	'orrc': 'Det NP_CP_null compobj_null Det NP Vt_act Vt_act Det NP eos',
# 	'rrc': 'Det NP_VoiceP Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'frc': 'Det NP_CP compsubj aux_pass Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'progrrc': 'Det NP_ProgP ProgP Vt_pass Prep Det NP Vt_act Det NP eos',
# 	'progrc': 'Det NP_ProgP compsubj aux_prog ProgP Vt_pass Prep Det NP Vt_act Det NP eos',
# 	# 'amv': 'Det NP Vt_act Det NP conj aux Adj',
# 	# 'amv2': 'Det NP Vt_act Det NP conj Vt_act Det NP',
# 	'amv1': 'Det NP Vt_act Det NP eos',
# 	'amv2': 'Det NP aux Adj eos',
# 	'amv3': 'Det NP Vt_act Prep Det NP eos',
# 	'amv4': 'Det NP Vt_act Det NP conj Vt_act Det NP eos'
# 	}




