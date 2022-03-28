import re
import math
import random
from copy import deepcopy
import numpy as np
import pickle

from actr_model import actr_model
import functions2


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




with open('./declmem/lexical_chunks_ep.pkl', 'rb') as f:
	lexical_chunks_ep = pickle.load(f)

with open('./declmem/lexical_chunks_wd.pkl', 'rb') as f:
	lexical_chunks_wd = pickle.load(f)

with open('./declmem/syntax_chunks_ep.pkl', 'rb') as f:
	syntax_chunks_ep = pickle.load(f)

with open('./declmem/syntax_chunks_wd.pkl', 'rb') as f:
	syntax_chunks_wd = pickle.load(f)

with open(train_fname,'r') as f:
	train_dat = f.readlines()


train_dat = [x.strip() for x in train_dat]


# num_sents = 5000
num_sents = 1000
decay = 0.5
max_activation = 1.5
prior_weight = 1
# time_factor = 0.3
time_factor = 0.05

## Train EP

for i in range(30):
	random.seed(i)
	np.random.seed(i)

	curr_train_dat = deepcopy(train_dat)
	random.shuffle(curr_train_dat)

	curr_train_dat = curr_train_dat[:num_sents]

	# Train EP
	actr_model_ep = actr_model(decay,
							  max_activation,
							  prior_weight,
							  time_factor,
							  syntax_chunks_ep,
							  lexical_chunks_ep,
							  functions2.supertag_sentence_ep)

	actr_model_ep.update_counts(curr_train_dat)
	actr_model_ep.update_base_activation()
	actr_model_ep.update_lexical_activation()

	ep_fname = './trained_models/ep_train%sk_tf%s_seed%s.pkl'%(str(num_sents//1000), str(time_factor), i)

	with open(ep_fname, 'wb') as f:
		pickle.dump(actr_model_ep, f)



	actr_model_wd = actr_model(decay,
						  max_activation,
						  prior_weight,
						  time_factor,
						  syntax_chunks_wd,
						  lexical_chunks_wd,
						  functions2.supertag_sentence_wd)


	# Train WD

	actr_model_wd.update_counts(curr_train_dat)
	actr_model_wd.update_base_activation()
	actr_model_wd.update_lexical_activation()


	wd_fname = './trained_models/wd_train%sk_tf%s_seed%s.pkl'%(str(num_sents//1000), str(time_factor), i)

	with open(wd_fname, 'wb') as f:
		pickle.dump(actr_model_wd, f)


	if i%5 == 0:
		print('Trained %s models'%str(i+1))









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




