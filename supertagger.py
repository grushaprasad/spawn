import re
import math
import random
from copy import deepcopy
import numpy as np
import pickle
import ccg


"""
To do: 

- Change "give up" mechanism. Try for max iters once. If that doesn't work, then give up  on priors and keep trying for max iters till parse is reached. (Instead of right now where )

- Maybe specify max iters in terms of time?? 



"""

def supertag_sentence(actr_model, sentence, use_priors = True, print_stages=False, end_states=[{'left': 'end', 'right': '', 'combinator': ''}]):
	"""
	Input:
	  - actr_model (initialized with a specific grammar, activations etc)
	  - sentence (string)
	  - use_priors (Boolean): Should prior knowledge influence tag retrieval
	  - print_stages (Boolean): should stages of parsing be printed
	  - end_states (list): all possible valid end states

	Output:
	  - parse_states (list) final parse states at the end of each word
	  - supertags (list) final tags for each word (and null words)
	  - words (list) words in the sentence + added null elements
	  - act_vals (list of list) each sublist has the activation for every tag that was retrieved. -- will be used to compute time taken to parse the sentence. 
	"""
	# initialize variables
	words = sentence.split()  
	tr_rules = actr_model.type_raising_rules
	null_words = actr_model.null_mapping.values()

	supertags = []
	act_vals = [[] for word in words] #activation values for all tags considered.
	parse_states = [None] 
	inhibition = {}
	i = 0
	max_iters = actr_model.max_iters # number of iterations before "giving up"
	num_iters = 0
	curr_time = 0 #keep track of each retrieval
	# use_priors = True # 			  

	while(i < len(words) and num_iters < max_iters):
		# if num_iters > (max_iters/2): #ignore priors if things are taking too long
		# 	use_priors = False
		num_iters +=1
		if print_stages:
			print()
			print('curr_word', words[i])
			print('words', words)
			print('supertags', supertags)

		#get current word and tags that can be associated with the word
		curr_word = words[i] 
		poss_tags = deepcopy(actr_model.lexical_chunks[curr_word]['syntax'])

		found_valid_tag=False

		j = 0

		while len(poss_tags) > 0: # while there is a tag that hasn't been considered
			goal_buffer = (i, parse_states[-1])

			#generate supertag 
			curr_tag, curr_act = generate_supertag(actr_model, goal_buffer, curr_word, inhibition, poss_tags, curr_time, use_priors=use_priors)

			curr_tag_chunk = actr_model.syntax_chunks[curr_tag]

			# update variables
			curr_time += actr_model.convert_to_rt([curr_act])  ## should I add 50 ms for production rule firing? (see model.py)
			act_vals[i].append(curr_act) 

			# try to combine current tag with current parse state
			combined = ccg.combine(tag = curr_tag_chunk, parse_state =  goal_buffer[-1], tr_rules = tr_rules)

			goal_buffer = (i, combined) # update parse state

			if print_stages:
				print('retrieved tag', curr_tag)
				print(ccg.convert_to_rule(goal_buffer[-1]),ccg.convert_to_rule(curr_tag_chunk), ccg.convert_to_rule(combined))

			if combined: #i.e., not None
				found_valid_tag = True
				break #stop looking for more tags
			else:
				poss_tags.remove(curr_tag) #discard current tag

		# For the last word check if combined state is a valid end state
		if i == len(words)-1 and len(end_states) > 0 and combined not in end_states:
			found_valid_tag = False

		if found_valid_tag:
			supertags.append(curr_tag)
			parse_states.append(combined)

			# Is the next token a null element?
			inhibit_key = (i,ccg.convert_to_rule(combined))
			curr_inhibition = inhibition.get(inhibit_key, [])

			null_el = predict_null(actr_model, curr_word, curr_tag, curr_inhibition, curr_time, use_priors=use_priors) 

			if null_el: # if it is null, add the null token to the sequence of words
				words.insert(i+1, null_el) 
				act_vals.insert(i+1, [])

			i+=1 #go to next token
		else: 
			# decide which word to reanaluse
			if actr_model.reanalysis_type == 'start': # always go back to beginning
				reanalyze_ind = 0
			elif actr_model.reanalysis_type == 'prev_word': # always go back one word
				reanalyze_ind = i-1
			else: # sample based on uncertainty
				reanalyze_ind = get_reanalyze_ind(actr_model, supertags, words)
			
			# Remove predicted tags and parse states for re-analyzed words
			num_to_pop = i - reanalyze_ind
			for k in range(num_to_pop):
				removed_tag = supertags.pop()
				removed_parse_state = parse_states.pop() 


				# add inhibition to all rules along the way
				
				key = (i-1,ccg.convert_to_rule(parse_states[-1]))
				inhibition[key] = inhibition.get(key, [])
				inhibition[key].append({'tag': removed_tag, 'time': curr_time})

				if words[i] in null_words: #e.g., 'comp_del'
					removed_null = words.pop(i)
					inhibition[key].append({'tag': removed_null, 'time': curr_time}) #used in predict_null
					del act_vals[i]
				else: #add inhibition to the non-null choice
					inhibition[key].append({'tag': 'not-null', 'time': curr_time}) #used in predict_null

				i -= 1 # go back one word
			
			#make sure i is reset to reanalyze_ind	
			assert i == reanalyze_ind  #will go back to reanalysis ind in next loop
	
	if num_iters >= max_iters:
		return(None, None, None, None) # "give up"
	else:
		return(parse_states, supertags, words, act_vals)



def get_entropy(d, temp):
	"""
	Input: 
		- d: dictionary. Keys are categories, values are activation. 
		- temp: float. temperature for softmax
	Output: entropy of the dictionary (as measured using activation)
	"""
	vals = np.array(list(d.values()))
	probs = np.exp(vals/temp)/sum(np.exp(vals/temp))
	ent = -np.sum(probs*np.log(probs))

	return ent

def weighted_sample(d):
	"""
	Input: dictionary. Keys are categories, values are weights. 
	Output: a key sampled by the weight
	"""
	
	probs = np.array(list(d.values()))/sum(d.values())
	pick = np.random.choice(list(d.keys()), p = probs)
	return pick

def get_reanalyze_ind(actr_model, supertags, words):
	"""
	Input: 
		supertags (list): supertags for currently processed words
		words (list): all words (null tokens included)
	Output:
		rand_ind (int): index corresponding to one of the previous words
	"""
	options = {}
	for i in range(len(supertags)): #for each supertag generated (i.e., words processed)
		competitors = actr_model.lexical_chunks[words[i]]['syntax'] # number of alternative supertags

		act_dict = {}

		for comp in competitors:
			act = actr_model.eps ## avoid zero activation
			act += actr_model.base_act[comp] 
			act += actr_model.lexical_act[words[i]][comp]
			act_dict[comp] = act

		entropy = get_entropy(act_dict, actr_model.temperature)
		options[i] = entropy
		
		# options[i] = len(competitors) ## TO DO: Make this real entropy

	rand_ind = weighted_sample(options) 
	return(rand_ind)

def predict_null(actr_model, word, curr_tag, curr_inhibition, curr_time, print_prob=False, use_priors=True):
	"""
	Input:
	  word (string): word being processed
	  curr_tag (string): tag being considered for the word
	  curr_inhibition (list): inhibition associated with current parse state and word pos
	  curr_time (float): time since beginning of sentence
	  print_prob (boolean): True = print probability of null vs not-null
	  use_priors (boolean): True = use lexical and base level activations

	Output:
	  choice: string of null element or None
	"""

	if curr_tag in actr_model.null_mapping:
		if use_priors:
			curr_act_dict = {tag:val for tag,val in actr_model.lexical_null_act[word].items()}
		else: 
			curr_act_dict = {tag:0 for tag in actr_model.lexical_null_act[word]}

		# add in inhibition
		for tag in curr_act_dict:
			curr_act_dict[tag]-= actr_model.compute_inhibition(tag, curr_inhibition, curr_time)

		# add in noise
		for tag in curr_act_dict:  
			curr_act_dict[tag] += np.random.normal(0, actr_model.noise_sd)

		if print_prob:
			print(word, curr_tag)
			print(curr_inhibition)
			print(curr_act_dict)
			print(curr_time)

		choice = max(curr_act_dict, key=lambda key:curr_act_dict[key])
		if choice != 'not-null':
			return choice
	

def generate_supertag(actr_model, goal_buffer, curr_word, inhibition, poss_tags, curr_time, use_priors = True):
	"""
	Input:
	  goal_buffer (tuple): (position, parse_state)
	  curr_word (string)
	  inhibition (dictionary): key = parse_state + pos
	                           val = list of previously tried keys that did not work 
	  poss_tags (list): list of possible tags associated with the word
	  curr_time (float): time since sentence start
	  use_priors (boolean): True = use lexical and base level activations

	Output:
	  supertag (str): retrieved tag
	  act (float): activation value of the retrieved tag

	"""
	curr_act_dict = {tag:0 for tag in poss_tags}

	if use_priors:
		# Add in base level activation
		for tag in curr_act_dict:
			curr_act_dict[tag] = actr_model.base_act[tag]

	    # Add in activation from the word
		for tag in curr_act_dict:
			curr_act_dict[tag] += actr_model.lexical_act[curr_word][tag]  

	
	# Add in inhibition from the goal_buffer for things that did not work
	goal_buffer_str = (goal_buffer[0], ccg.convert_to_rule(goal_buffer[1]))
	
	if goal_buffer_str in inhibition: 
		curr_inhibition = inhibition[goal_buffer_str]

		for tag in curr_act_dict:
			tag_inhibition = actr_model.compute_inhibition(tag, curr_inhibition, curr_time)
			curr_act_dict[tag] -= tag_inhibition
    
    # Add in random noise

	for tag in curr_act_dict:  
		curr_act_dict[tag] += np.random.normal(0, actr_model.noise_sd)

	supertag = max(curr_act_dict, key=lambda key:curr_act_dict[key])
	act_val = curr_act_dict[supertag]


	return(supertag, act_val)


def print_states(tr_rules, tag_list, word_list):
	goal_buffer = None
	for i,tag_label in enumerate(tag_list):
		tag = syntax_chunks_wd2[tag_label]
		print()
		print(word_list[i])
		print('goal_buffer before', goal_buffer)
		print('tag', tag)

		goal_buffer = ccg.combine(tr_rules = tr_rules, tag = tag, goal_buffer = goal_buffer)
		
		# print('word,tag', word_list[i], tag)
		# print('goal_buffer after', goal_buffer)
		curr_rule = tag['left'] + tag['combinator'] + tag['right']
		curr_state = goal_buffer['left'] + goal_buffer['combinator'] + goal_buffer['right']
		print(word_list[i], curr_rule, curr_state)



## Test the supertagger on randomly initialized models

if __name__ == '__main__':
	from model import actr_model

	# fname = './data/train10000.txt'
	fname = 'test_sents_supertagger.txt'
	models = ['EP', 'WD', 'WD2']
	# models = ['WD2']
	print_tags = True
	# print_tags = False
	print_stages = True
	# print_stages = False

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

	global_sd = 1

	# Hyperparameters that do not change
	decay = 0.5           # Standard value for ACT-R models
	max_activation = 1.5  # Standard value for ACT-R models
	latency_exponent = 1  # Will always set this to 1 (because then it matches eqn 4 from Lewis and Vasisth)
	noise_sd = np.random.uniform(0.2,0.5)
	latency_factor = np.random.beta(2,6)
	max_iters = 1000
	reanalysis_type = 'start'
	temperature = 0

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
							  reanalysis_type,
							  temperature)

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
							  reanalysis_type,
							  temperature)

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
							  reanalysis_type,
							  temperature
							  )

	# print(actr_model_wd.lexical_null_act['defendant'])

	# partial_states = [{'left': 'DP', 'right': 'PP', 'combinator': '/'}, {'left': 'TP', 'right': 'DP', 'combinator': '/'}, {'left': 'end', 'right': '', 'combinator': ''}]
	partial_states = []
	with open(fname) as f:
		for line in f:
			print(line)
			if 'WD' in models:
				parse_states, supertags, words, act_vals = supertag_sentence(actr_model_wd, line, end_states=partial_states, print_stages=False)
				while parse_states == None:
					parse_states, supertags, words, act_vals = supertag_sentence(actr_model_wd, line, end_states=partial_states, print_stages=False)
				if print_tags: print('WD', supertags)

			if 'WD2' in models:
				parse_states, supertags, words, act_vals = supertag_sentence(actr_model_wd2, line, end_states=partial_states, print_stages=print_stages)
				while parse_states == None:
					parse_states, supertags, words, act_vals = supertag_sentence(actr_model_wd2, line, end_states=partial_states, print_stages=print_stages)
				if print_tags: print('WD2', supertags)

			if 'EP' in models:
				parse_states, supertags, words, act_vals = supertag_sentence(actr_model_ep, line, end_states=partial_states, print_stages=False)
				while parse_states == None:
					parse_states, supertags, words, act_vals = supertag_sentence(actr_model_ep, line, end_states=partial_states, print_stages=False)
				if print_tags: print('EP', supertags)
			print()

