import numpy as np 
import create_declmem
import os
import random

## For now I will not worry about plausibility 

## Types of MVs

# Simple transitive: det N verb det N
# Simple instransitive: det N verb (adv)
# Simple transitive loc: det N verb prep det N
# simple aux: det N verb aux Adj

# Maybe for now I will sample all of these equally. 

## Any RC can get one of these MV types

# All my RCs are going to modify the subject for now. 

seed = 7

random.seed(seed)
np.random.seed(seed)

num_sents = 10000

fname = './data/train%s.txt'%(str(num_sents))

if not os.path.isdir('./data/'):
	os.makedirs('./data/')

def generate_amv_trans(subj = None):
	if subj == None:
		subj = get_subj()
	obj = get_obj()
	verb = get_trans_verb()

	sent = ' '.join([subj, verb, obj])

	return(sent)


def generate_amv_transloc(subj = None):
	if subj == None:
		subj = get_subj()
	loc = get_loc()
	verb = get_transloc_verb()

	sent = ' '.join([subj, verb, loc])
	return(sent)


def generate_amv_intrans(subj = None):
	if subj == None:
		subj = get_subj()
	verb = get_intrans_verb()  #samples from ambig and unambig

	sent = ' '.join([subj, verb])
	return(sent)


def generate_amv_aux(subj = None):
	if subj == None:
		subj = get_subj()
	verb = get_aux()
	adj = np.random.choice(create_declmem.adjectives)

	sent = ' '.join([subj, verb, adj])
	return(sent)


def generate_amv_conj(subj = None):
	if subj == None:
		subj = get_subj()

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type1 = np.random.choice(mvs, p=mv_probs)

	sent1 = function_dict[mv_type1](subj=subj)

	mv_type2 = np.random.choice(mvs, p=mv_probs)

	sent2 = function_dict[mv_type2](subj='')

	sent = ' '.join([sent1, 'and', sent2])

	return(sent)


def generate_src():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_subj, 'who', rc_verb, rc_obj])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)

	sent = function_dict[mv_type](subj=rc)
	return(sent)


def generate_orc():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_obj, 'who', rc_subj, rc_verb])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)

	sent = function_dict[mv_type](subj=rc)
	return(sent)

def generate_orrc():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_obj, rc_subj, rc_verb])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)

	sent = function_dict[mv_type](subj=rc)
	return(sent)

def generate_frc():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_subj, 'who was', rc_verb, 'by', rc_obj])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)
	sent = function_dict[mv_type](subj=rc)
	return(sent)

def generate_rrc():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_subj, rc_verb, 'by', rc_obj])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)

	sent = function_dict[mv_type](subj=rc)
	return(sent)

def generate_progrc():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_subj, 'who was being', rc_verb, 'by', rc_obj])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)

	sent = function_dict[mv_type](subj=rc)
	return(sent)


def generate_progrrc():
	rc_subj = get_subj()
	rc_verb = get_ambig_verb()
	rc_obj = get_obj()

	rc = ' '.join([rc_subj, 'being', rc_verb, 'by', rc_obj])

	mvs = ['amv_trans', 'amv_intrans', 'amv_transloc', 'amv_aux', 'amv_conj']
	mv_probs = [1/3, 1/4, 1/12, 1/4, 1/12]

	mv_type = np.random.choice(mvs, p=mv_probs)

	sent = function_dict[mv_type](subj=rc)
	return(sent)


def get_subj():
	det = np.random.choice(create_declmem.determiners)
	noun = np.random.choice(create_declmem.animate_nouns)

	has_adj = np.random.choice([1,2,3,4,6,7,8])
	if has_adj == 1 :
		adj = np.random.choice(create_declmem.adjectives)
	else:
		adj = ''

	return(' '.join([det, adj, noun]))


def get_obj():
	det = np.random.choice(create_declmem.determiners)
	noun = np.random.choice(create_declmem.obj_nouns)

	has_adj = np.random.choice([1,2,3,4,6,7,8])
	if has_adj == 1 :
		adj = np.random.choice(create_declmem.adjectives)
	else:
		adj = ''

	return(' '.join([det, adj, noun]))


def get_loc():
	prep = np.random.choice(create_declmem.prepositions)
	det = np.random.choice(create_declmem.determiners)
	noun = np.random.choice(create_declmem.location_nouns)

	has_adj = np.random.choice([1,2,3,4,6,7,8])
	if has_adj == 1 :
		adj = np.random.choice(create_declmem.adjectives)
	else:
		adj = ''

	return(' '.join([prep, det, adj, noun]))


def get_ambig_verb():
	verb = np.random.choice(create_declmem.ambig_verbs)
	return(verb)

def get_trans_verb():
	verb = np.random.choice(create_declmem.trans_verbs)

	has_adv = np.random.choice([1,2,3,4,6,7,8,9,10])
	if has_adv == 1 :
		adv = np.random.choice(create_declmem.adverbs)
	else:
		adv = ''

	return(' '.join([verb, adv]))

def get_transloc_verb():
	verb = np.random.choice(create_declmem.transloc_verbs)
	# prep = np.random.choice(create_declmem.prepositions)
	# location = np.random.choice(create_declmem.location_nouns)

	has_adv = np.random.choice([1,2,3,4,6,7,8,9,10])
	if has_adv == 1 :
		adv = np.random.choice(create_declmem.adverbs)
	else:
		adv = ''

	return(' '.join([verb, adv]))


def get_intrans_verb():
	verb = np.random.choice(create_declmem.intrans_verbs)

	has_adv = np.random.choice([1,2,3,4])
	if has_adv == 1 :
		adv = np.random.choice(create_declmem.adverbs)
	else:
		adv = ''

	return(' '.join([verb, adv]))

def get_aux():
	verb = np.random.choice(create_declmem.aux)
	return(verb)


def generate_sentence():
	struc = np.random.choice(list(prob_dict.keys()), p = list(prob_dict.values()))

	sent = function_dict[struc]()
	# sent = function_dict['amv_conj']()

	return(sent)



function_dict = {
	'src': generate_src,
	'orc': generate_orc,
	'orrc': generate_orrc,
	'frc': generate_frc,
	'rrc': generate_rrc,
	'progrrc': generate_progrrc,
	'progrc': generate_progrc,
	'amv_trans': generate_amv_trans,
	'amv_intrans': generate_amv_intrans,
	'amv_transloc': generate_amv_transloc,
	'amv_aux': generate_amv_aux,
	'amv_conj': generate_amv_conj
}

# src_prob = (14182 + 15024 + 18229)/3000000
# orc_prob = (2943 +  1976 + 1802)/3000000
# orrc_prob = (5455 + 4746 + 3385)/3000000
# frc_prob = (77993 + 882 + 375)/3000000
# rrc_prob = (26841 + 3302 + 3918)/3000000
# prog_prob = (13567 + 150 + 129)/3000000
# amv_prob = 1 - (src_prob + orc_prob + orrc_prob + frc_prob + rrc_prob + prog_prob)


src_prob = (14182 + 15024 + 18229)/3000000
orc_prob = (2943 +  1976 + 1802)/3000000
orrc_prob = (5455 + 4746 + 3385)/3000000
frc_prob = (3188 + 2867 + 1224)/3000000 # 0.0005
rrc_prob = (10730 + 10733 + 12788)/3000000 # 0.01
# prog_prob = (542 + 488 + 421)/3000000
prog_prob = frc_prob*2 #based on the fact that frc didn't show up
amv_prob = 1 - (src_prob + orc_prob + orrc_prob + frc_prob + rrc_prob + prog_prob)


prob_dict = {
	'src': src_prob,
	'orc': orc_prob,
	'orrc': orrc_prob,
	'frc': frc_prob,
	'rrc': rrc_prob,
	# 'progrrc': prog_prob/2,
	# 'progrc': prog_prob/2,
	'progrrc': prog_prob*0.95, #based on relation between frc and rrc
	'progrc': prog_prob*0.05,
	'amv_trans': amv_prob/3,
	'amv_intrans': amv_prob/4,
	'amv_transloc': amv_prob/12,
	'amv_aux':amv_prob/4,
	'amv_conj': amv_prob/12
}

#print(generate_sentence())


with open(fname, 'w') as f:
	for i in range(num_sents):
		f.write(generate_sentence() + ' .')
		f.write('\n')
# for struc in prob_dict:
# 	print(function_dict[struc]())




