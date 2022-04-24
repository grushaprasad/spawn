import re
# import declmem
import math
import random
from copy import deepcopy
import numpy as np
import pickle

import supertagger
# import train
import csv

from actr_model import actr_model


# random.seed(7)
# np.random.seed(7)

print('HELLO FROPPY')

# train_fname = './trained_models/ep_trained_2M.pkl'

# train_fname_ep = './trained_models/ep_trained_1k_3amv.pkl'
# train_fname_wd = './trained_models/wd_trained_1k_3amv.pkl'

# ntrain_sents = 1000


# train_fname_ep = './trained_models/ep_train1k_'
# train_fname_wd = './trained_models/wd_train1k_'


# prime_fname_ep = './predictions/ep_train1k.csv'
# prime_fname_wd = './predictions/wd_train1k.csv'


# prime_fname_ep = './predictions/ep_train1k.csv'
# prime_fname_wd = './predictions/wd_train1k.csv'


# sd = 0.5
sd = 1
num_parts = 512

# fname = './trained_models/wd_train0.1k_sd1.5_part26.pkl'
# with open(fname, 'rb') as f:
# 	actr_model_wd = pickle.load(f)


# print(actr_model_ep.lexical_act['examined'])

# print(actr_model_ep.base_act['Vt_pass'])
# print(actr_model_ep.base_act['Vt_act'])

# num_passive = 0

# for i in range(50):
# 	goal_buffer, supertags, words, act_vals = actr_model_wd.supertag_sentence(actr_model_ep, 'the girl being scratched by the cat enjoyed the sunny afternoon .')	

# 	if supertags[-1] == 'Vt_pass':
# 		num_passive += 1

# print(num_passive)

# with open(train_fname_wd, 'rb') as f:
# 	actr_model_wd = pickle.load(f)


# print('CHECKING PARSE OF RELEVANT SENTENCES')

sents = [
		 'the lawyer examined the defendant .',
		 'the lawyer arrived at the palace .',
		 'the lawyer sang beautifully .',
		 'the lawyer examined by the defendant loved the unreliable cat .',
		 'the lawyer examined by the defendant was unreliable .',
		 'the lawyer examined by the defendant arrived at the palace .',
		 'the lawyer examined by the defendant sang beautifully .',
		 'the lawyer being examined by the defendant loved the unreliable cat .',
		 'the lawyer being examined by the defendant was unreliable .',
		 'the lawyer being examined by the defendant arrived at the palace .',
		 'the lawyer being examined by the defendant sang beautifully .',
		 'the lawyer who was examined by the defendant loved the unreliable cat .',
		 'the lawyer who was examined by the defendant was unreliable .',
		 'the lawyer who was examined by the defendant arrived at the palace .',
		 'the lawyer who was examined by the defendant sang beautifully .',
		 'the lawyer who examined the defendant loved the unreliable cat .',
		 'the lawyer who examined the defendant was unreliable .',
		 'the lawyer who examined the defendant arrived at the palace .',
		 'the lawyer who examined the defendant sang beautifully .',
		 'the lawyer who the defendant examined loved the unreliable cat .',
		 'the lawyer who the defendant examined was unreliable .',
		 'the lawyer who the defendant examined arrived at the palace .',
		 'the lawyer who the defendant examined sang beautifully .',
		 'the lawyer the defendant examined loved the unreliable cat .',
		 'the lawyer the defendant examined was unreliable .',
		 'the lawyer the defendant examined arrived at the palace .',
		 'the lawyer the defendant examined sang beautifully .',
		 'the lawyer examined the defendant and loved the cat .',
		 'the lawyer examined the defendant and was unreliable .',
		 'the lawyer examined the defendant and arrived at the palace .',
		 'the lawyer examined the defendant and sang beautifully .',
		]
with open('./declmem/syntax_chunks_wd2.pkl', 'rb') as f:
	syntax_chunks_wd2 = pickle.load(f)

with open('./declmem/lexical_chunks_wd2.pkl', 'rb') as f:
	lexical_chunks_wd2 = pickle.load(f)

with open('./declmem/syntax_chunks_ep.pkl', 'rb') as f:
	syntax_chunks_ep = pickle.load(f)

with open('./declmem/lexical_chunks_ep.pkl', 'rb') as f:
	lexical_chunks_ep = pickle.load(f)

# decay = 0.5           # Standard value for ACT-R models
# max_activation = 1.5  # Standard value for ACT-R models
# latency_exponent = 1  # Will always set this to 1 (because then 
# noise_sd = 0.5
# latency_factor = 0.4


# actr_model_ep =  actr_model(decay,
# 						max_activation,
# 						noise_sd,
# 						latency_factor,
# 						latency_exponent,
# 						syntax_chunks_ep,
# 						lexical_chunks_ep,
# 						supertagger.supertag_sentence)

# for sent in sents:
# 	print(sent)
# 	print('EP', actr_model_ep.supertag_sentence(actr_model_ep, sent)[1])

# 	goal_buffer, supertags, words, act_vals = actr_model_ep.supertag_sentence(actr_model_ep, sent)
# 	act_vals = [[round(x, 2) for x in item] for item in act_vals]

# 	actr_model_ep.update_counts([sent])
# 	actr_model_ep.update_base_activation()
# 	actr_model_ep.update_lexical_activation()
# 	print('EP', supertags)
# 	print(act_vals)
# 	print(actr_model_ep.time)
	
# 	print()



# with open('./data/stimuli/list_1A.txt', 'r') as f:
# 	stims = f.readlines()

# stims = [x.lower() for x in stims]
# stims = [x.strip() for x in stims]


## Generating priming predictions for EP account



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

			final_state, tags, words, act_vals = model.supertag_sentence(model, sent, partial_states=partial_states)

			final_state_rule = final_state['left'] + final_state['combinator'] + final_state['right']

			if tags[-1] == 'Vt_pass':
				passive = 1
			else:
				passive = 0

			preds.append([stim_fname,part_id,sent_id,passive, final_state_rule])

	return(preds)

def flatten(l):
	return([x for sublist in l for x in sublist])

def compare_baseact_rrc_progrrc(model_fname, stim_fname, part_id):
	preds = []
	with open(stim_fname, 'r') as sf:
		stims = sf.readlines()

	stims = [x.lower() for x in stims]
	stims = [x.strip() for x in stims]

	with open(model_fname, 'rb') as mf:
		model = pickle.load(mf)

	delta_baseact_rrc = []
	delta_baseact_progrrc = []

	for sent_id,sent in enumerate(stims):
		if len(sent.split()) > 3:
			time_before = model.time
			baseact_before = model.base_act['NP_CP_null']
			#final_state, tags, words, act_vals = model.supertag_sentence(model, sent)
			model.update_counts([sent])
			model.update_base_activation()

			baseact_after = model.base_act['NP_CP_null']
			time_after = model.time
			words = sent.split()
			if words[2] == 'being':
				delta_baseact_progrrc.append(baseact_after-baseact_before)
			if words[3] == 'by':
				delta_baseact_rrc.append(baseact_after-baseact_before)
			#print(sent.split()[2]+ ' ' + sent.split()[3], round(time_after-time_before))

			#print(tags[3], len(act_vals[3]), len(flatten(act_vals)))
		else:
			continue

	return(delta_baseact_rrc, delta_baseact_progrrc)


def save_preds(preds, fname):
	with open(fname, 'w') as f:
		writer = csv.writer(f, delimiter = ',')
		writer.writerow(['list', 'part_id', 'sent_id', 'passive', 'goalstate'])

		for pred in preds:
			writer.writerow(pred)


stim_fnames = []

for a in ['1','2','3','4']:
	for b in ['A', 'B', 'C', 'D']:
		for r in ['', '_rev']:
			stim_fnames.append('./data/stimuli/list_' + a + b + r + '.txt')


ep_preds = []
wd_preds = []
wd2_preds = []

# for num_sents in [100, 1000, 10000]:
#for num_sents in [100, 1000]:
for num_sents in [100, 500]:
#for num_sents in [10000]:
	print('---------------')
	print(num_sents)

	for i in range(num_parts):

		curr_stim_fname = stim_fnames[i%32]

		# EP preds
		random.seed(i)
		np.random.seed(i)

		ep_model_name = './trained_models/ep_train%sk_sd%s_part%s.pkl'%(str(num_sents/1000), str(sd), str(i))

		# with open(ep_model_name, 'rb') as mf:
		# 	model = pickle.load(mf)

		# comp_types = model.lexical_chunks['comp_del']['syntax']
		# for comp in comp_types:
		# 	print(comp,  model.base_count[comp],model.base_instance[comp],model.base_act[comp])
		# 	print(model.time, model.compute_baseact(comp))
		# print('-----')


		curr_ep_preds = generate_priming_preds(ep_model_name, curr_stim_fname, i)
		ep_preds.extend(curr_ep_preds)


		# WD preds
		# random.seed(i)
		# np.random.seed(i)

		# wd_model_name = './trained_models/wd_train%sk_sd%s_part%s.pkl'%(str(num_sents/1000), str(sd), str(i))


		# curr_wd_preds = generate_priming_preds(wd_model_name, curr_stim_fname, i)
		# wd_preds.extend(curr_wd_preds)


		# WD2 preds
		random.seed(i)
		np.random.seed(i)

		wd2_model_name = './trained_models/wd2_train%sk_sd%s_part%s.pkl'%(str(num_sents/1000), str(sd), str(i))

		curr_wd2_preds = generate_priming_preds(wd2_model_name, curr_stim_fname, i)
		wd2_preds.extend(curr_wd2_preds)

		# with open(wd2_model_name, 'rb') as mf:
		# 	model = pickle.load(mf)

		# comp_types = model.lexical_chunks['comp_del']['syntax']
		# for comp in comp_types:
		# 	#print(comp, model.base_count[comp],model.base_instance[comp],model.base_act[comp],model.compute_baseact(comp))
		# 	print(comp)
		# 	time_seq = model.time - np.array(deepcopy(model.base_instance[comp]))

		# 	activation = np.log(sum(np.power(time_seq, -model.decay)))
		# 	print(time_seq)
		# 	print(activation)


			#print(model.time, model.compute_baseact(comp))

		# print('=========')



		if i%10 == 0:
			print('Processed %s participants'%str(i+1))


	prime_fname_ep = './predictions/ep_train%sk_sd%s.csv'%(str(num_sents/1000), str(sd))

	prime_fname_wd = './predictions/wd_train%sk_sd%s.csv'%(str(num_sents/1000), str(sd))

	prime_fname_wd2 = './predictions/wd2_train%sk_sd%s.csv'%(str(num_sents/1000), str(sd))

	save_preds(ep_preds, prime_fname_ep)

	# save_preds(wd_preds, prime_fname_wd)

	save_preds(wd2_preds, prime_fname_wd2)




### Understanding why ProgRRC has lower priming effect than 
# for num_sents in [100, 1000]:
# #for num_sents in [10000]:
# 	print('---------------')
# 	print(num_sents)
# 	means_rrc = []
# 	means_progrrc = []
# 	rrc_greater_than_progrrc = []

# 	for i in range(1):

# 		curr_stim_fname = stim_fnames[i%32]


# 		# WD preds
# 		random.seed(i)
# 		np.random.seed(i)

# 		wd_model_name = './trained_models/wd_train%sk_sd%s_part%s.pkl'%(str(num_sents/1000), str(sd), str(i))


# 		delta_baseact_rrc, delta_baseact_progrrc = compare_baseact_rrc_progrrc(wd_model_name, curr_stim_fname, i)

# 		mean_delta_baseact_rrc = sum(delta_baseact_rrc)/len(delta_baseact_rrc)
# 		mean_delta_baseact_progrrc = sum(delta_baseact_progrrc)/len(delta_baseact_progrrc)

# 		means_rrc.append(mean_delta_baseact_rrc)
# 		means_progrrc.append(mean_delta_baseact_progrrc)
# 		rrc_greater_than_progrrc.append(mean_delta_baseact_rrc > mean_delta_baseact_progrrc)


	
# 	print('RRC ', sum(means_rrc)/len(means_rrc))
# 	print('ProgRRC ', sum(means_progrrc)/len(means_progrrc))
# 	print('RRC > ProgRRC ', sum(rrc_greater_than_progrrc)/len(rrc_greater_than_progrrc))
		

				
# print('EP ACCOUNT')
# generate_priming_preds(prime_fname_ep, train_fname_ep, stim_fnames)
# print()


# print('WD ACCOUNT')
# generate_priming_preds(prime_fname_wd, train_fname_wd, stim_fnames)
# print()
				





# for sent in stims:
# 	print(sent)
# 	print('WD', actr_model_wd.supertag_sentence(actr_model_wd, sent)[1])

# 	print('EP', actr_model_ep.supertag_sentence(actr_model_ep, sent)[1])
	
# 	print()




# primes = {
# 	'rrc': 'the defendant examined by the lawyer was unreliable',
# 	'frc': 'the defendant who was examined by the lawyer was unreliable',
# 	'progrrc': 'the defendant being examined by the lawyer was unreliable',
# 	'src': 'the defendant who examined the lawyer was unreliable',
# 	'orrc': 'the defendant the lawyer examined was unreliable',
# 	'amv': 'the defendant examined the lawyer and was unreliable'
# }



# print('==========')
# print('MODEL PARAMETERS')
# print('----------')
# print('Decay', actr_model_ep.decay)
# print('Max activation', actr_model_ep.max_activation)
# print('Time factor', actr_model_ep.time_factor)
# print('Num training sents', ntrain_sents)

# print('==========')



# target = 'the patient examined'


# ## Generating priming predictions for EP account
# print('----------------------')
# print('EP account')
# print('----------------------')


# num_passive = 0


# for i in range(1000):
# 	final_state, tags, words = actr_model_ep.supertag_sentence(actr_model_ep, target)
# 	if tags[-1] == 'Vt_pass':
# 		num_passive+=1


# prop_passive_before = num_passive/1000

# print('Prop passive before:', prop_passive_before)

# for struc in primes:

# 	prime_sents = [primes[struc]]*3

# 	actr_model_ep.update_counts(prime_sents)
# 	actr_model_ep.update_base_activation()
# 	actr_model_ep.update_lexical_activation()

# 	num_passive = 0

# 	for i in range(1000):
# 		final_state, tags, words = actr_model_ep.supertag_sentence(actr_model_ep, target)

# 		if 'Vt_pass' in tags:
# 			num_passive+=1

# 	prop_passive_after = num_passive/1000

# 	print('Prop passive with ' + struc + ': ' + str(prop_passive_after))



# print()

# ## Generating priming predictions for WD account
# print('----------------------')
# print('WD account')
# print('----------------------')


# num_passive = 0


# for i in range(1000):
# 	final_state, tags, words = functions2.supertag_sentence_wd(actr_model_wd, target)

# 	if tags[-1] == 'Vt_pass':
# 		num_passive+=1


# prop_passive_before = num_passive/1000

# print('Prop passive before:', prop_passive_before)

# for struc in primes:

# 	prime_sents = [primes[struc]]*3

# 	actr_model_wd.update_counts(prime_sents)
# 	actr_model_wd.update_base_activation()
# 	actr_model_wd.update_lexical_activation()

# 	num_passive = 0

# 	for i in range(1000):
# 		final_state, tags, words = functions2.supertag_sentence_wd(actr_model_wd, target)

# 		if 'Vt_pass' in tags:
# 			num_passive+=1

# 	prop_passive_after = num_passive/1000

# 	print('Prop passive with ' + struc + ': ' + str(prop_passive_after))


## DEBUGGIN
# target = 'the lawyer the defendant examined loved the cat'
# target = 'the defendant who the lawyer'
# target = 'the defendant who was being loved by the lawyer loved the cat .'
# print(target)

# print(actr_model.lexical_chunks['being'])
# print('being', actr_model.lexical_act['being'])

# print(actr_model.lexical_chunks['was'])
# print('was', actr_model.lexical_act['was'])

# print('lawyer', actr_model.lexical_act['defendant'])


# print(functions2.supertag_sentence(actr_model, target)[1])

# print('----------------------')

# print(functions2.supertag_sentence(actr_model, target)[1])


# target = 'the lawyer the defendant'


# print('base act after')
# for key in ['Vt_pass', 'Vt_act']:
# 	print(key, actr_model.base_act[key])
# print()

# print('Assoc dict after')
# for key in ['Det', 'NP', 'aux', 'compsubj']:
# 	print(key, {key2: assoc_count[key][key2] for key2 in ['Vt_pass', 'Vt_act']})
# print()

# print('lexical dict after')
# for key in ['Vt_pass', 'Vt_act']:
# 	print(key, actr_model.lexical_count['examined'][key])
# print()


# print('lexical act after')
# for key in ['Vt_pass', 'Vt_act']:
# 	print(key, actr_model.lexical_act['examined'][key])
# print()



# num_passive = 0
# num_active = 0
# for i in range(1000):
# 	final_state, tags = functions2.supertag_sentence(actr_model, target)

# 	if tags[verb_ind] == 'Vt_pass':
# 		num_passive+=1

# 	if tags[verb_ind] == 'Vt_act':
# 		num_active+=1

# prop_passive_after = num_passive/1000
# prop_active_after = num_active/1000

# print('====================')
# print('TIME FACTOR: 0.05')
# print(prime_type)
# print('prop_passive', prop_passive_before, prop_passive_after)
# print('prop_active', prop_active_before, prop_active_after)

# for i in range(10):
# 	print(supertag_sentence('the defendant examined', base_count, base_act, assoc_count, lexical_count, max_activation)[0])

## Deleted stuff

# sent_list = [
# 			 'the defendant who was examined by the lawyer',
# 			 # 'the defendant examined by the lawyer',
# 			 'the defendant examined the lawyer',
# 			 # 'the defendant examined the lawyer',
# 			 # 'the defendant examined the lawyer'
# 			 ]

# tag_list = [
# 			'Det NP compsubj aux Vt_pass Prep Det NP',
# 			# 'Det NP Vt_pass Prep Det NP',
# 			'Det NP Vt_act Det NP',
# 			# 'Det NP Vt_act Det NP',
# 			# 'Det NP Vt_act Det NP'
# 			]


# print(basecount_dict)
# print()
# for key,val in lexical_dict.items():
# 	print(val)
# print()
# for key, val in assoc_dict.items():
# 	print(key, val)


# update_counts(base_count, base_instance, assoc_count, lexical_count, 1.5, sent_list, tag_list)

# decay = 0.5   

# print(compute_baseact('Det', base_count, base_instance, decay))
# print(compute_baseact('Vt_pass', base_count, base_instance, decay))


# print('base counts')
# print(base_count)
# print()


# print('base instance')
# print(base_instance['Vt_pass'])
# print()
# print('Lexical dict')
# for key,val in lexical_count.items():
# 	print(key,val)
# print()
# print('Assoc dict')
# for key, val in assoc_count.items():
# 	print(key, val)



# baseact_dict = {key:0.1 for key in declmem.syntax_chunks}

# temp = {key: 0 for key in declmem.syntax_chunks}
# assoc_dict = {key:temp for key in declmem.syntax_chunks}

# tag_buffer = ['Det', 'NP']


# for i,sent in enumerate(sents):
# 	num_pass = 0
# 	for j in range(1000):
# 		tags = supertag_sentence(sent, base_count, base_act, assoc_count, lexical_count, max_activation)
# 		if tags[verb_pos[i]][1] == 'Vt_pass':
# 			num_pass+=1
# 	print(sent)
# 	print('prop passive', num_pass/1000)
# 	print()

"""
Sequence of tags for "the defendant examined" under PP account

DP/NP NP (NP/NP)/PP

Do I want to assume that every syntax chunk is connected to every other syntax chunk? 
This seems implausible.. But maybe I can have higher weight connections between tags that are more likely to come after the other? 

"""

"""
syntax_chunk_dict = {
	rule1 : {rule1: val1, rule2: val1, rule3: val3 ...},
	rule2: {rule1: val1, rule2: val1, rule3: val3 ...}
}

val(x,y) is the prob that the rule 

"""


"""

Parsing algorithm  (Note: make sure I am implementing spreading activation correctly)

Read "the": 
 	- Retrieve lexical form
 	- Retrieve DP/NP and add to goal buffer
 		- Spreading activation from the lexical form. 

Read "defendant"
	- Retrieve lexical form
	- Retrieve NP and add to goal buffer

Read "who"
	- Retrieve lexical form 
	- Retrieve (NP\\NP)/(TP\\DP)

Read "was"
	- Retrieve lexical form
	- Retrieve (TP\\DP)/(NP\\NP)

Read "examined"
	- Retrieve lexical form
	- Retrieve (NP\\NP)/PP and add to goal buffer
		- Spreading activation from lexical form, and also from NP ? 
	- Combine NP and (NP\\NP)/PP to get NP/PP

Read "by"
	- Retrieve lexical form
	- Retrieve PP/DP
		- Spreading activation from lexical form and NP/PP

Read "the"
	- Retrieve lexical form
	- Retrieve DP/NP
		- Spread activation from PP/DP

Read "lawyer"
	- retrieve lexical form
	- Retrieve NP
		- Spread activation from 

"""


"""

What I want to think of as cues are all the things that are missing from the current goal buffer? This makes sense if you think you are looking to complete something? 

And maybe I can think of this as a bi-directional link? This is what explains why NP can spread activation to (NP\\NP)/PP ? 

"""


"""

The way bigram or trigram tags will affect things is by the tags that are formed in the buffer


"""

"""

Can I think of spreading activation as just sampling a value from a list? 

e.g., sing is a feature. This will spread activation to all lexical items that have this feature. 
So I can say sing: [n1, n2, n3 ... nx]

Ohh its not sampling. Its actually that each element in the list gets activation of:

S - ln(len(list))

Where S is maximum associative strength. It is set to 1.5 in many papers. 
"""