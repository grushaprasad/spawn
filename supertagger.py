import re
# import declmem_v2
import math
import random
from copy import deepcopy
import numpy as np
import pickle
import ccg



def convert_to_rule(state):
	if state != None:
		return(state['left'] + state['combinator'] + state['right'])


"""
Supertagging algorithm:

For every word:
	Generate supertag for the word. 
	While the supertag cannot be combined with the curent parse state:
		Generate a new supertag
	if supertag can be combined with current parse state:
		Process next word
	otherwise:
		Go back to previous word (but excluding the previously considered tag)
"""

"""
What needs to happen with reanalysis:

When I am reanalyzing, I need to keep track of all possible combinations of two words. 

Lets say w3 has A,B,C and w2 has D, E, F

w2 = D: (select from D,E,F)
	None of A,B or C work. 

w2 = E: (select from E,F)
	None of A,B or C work. 

w3 = F: (select from F)
	None of A,B or C work. 

Go back to w1. 

Success of reanalysis isn't if the current word can combine with the new tag. But instead it is if you are able to move past the point you were stuck at originally. 

"""

"""

Ideas about memory:

The notion of "going back but not repeating mistakes" assumes that there is some memory of previous parsing decisions. 
How is this memory encoded? 

DP/NP NP V_act PP??
DP/NP NP V_act
DP/NP NP_CP compprog_null Vt_act


"""
## TO DO: Fix the inhibition thing because it is not enough to get the right parses on RC sentences during training.. (right now its crashing at the first rrc)

def supertag_sentence(actr_model, sentence, print_stages=False, end_states=[{'left': 'end', 'right': '', 'combinator': ''}]):
	"""

	"""
	words = sentence.split() 
	tr_rules = actr_model.type_raising_rules
	null_words = actr_model.null_mapping.values()

	#initialize variables
	#supertags = ['' for word in words]
	supertags = []
	act_vals = [[] for word in words] #activation values for all tags considered.
	parse_states = [None] 
	inhibition = {}
	i = 0
	max_iters = actr_model.max_iters 
	num_iters = 0
	curr_time = 0
	use_priors = True  

	# while(i < len(words)):
	while(i < len(words) and num_iters < max_iters):
		if num_iters > (max_iters/2): #ignore priors if things are taking too long
			use_priors = False
		num_iters +=1
		if print_stages:
			print()
			print('curr_word', words[i])
			print('words', words)
			print('supertags', supertags)
			# print('state before', parse_states[-1])
		 #or should this be parse_states[i] ?
		curr_word = words[i]

		poss_tags = deepcopy(actr_model.lexical_chunks[curr_word]['syntax'])

		#max_tries = len(poss_tags)
		found_valid_tag=False

		j = 0

		# while j < max_tries:
		while len(poss_tags) > 0:
			goal_buffer = (i, parse_states[-1])
			curr_tag, curr_act = generate_supertag(actr_model, goal_buffer, curr_word, inhibition, poss_tags, curr_time, use_priors=use_priors)

			

			curr_time += actr_model.convert_to_rt([curr_act])  #keep track of how much time each retrieval takes

			act_vals[i].append(curr_act) #adds activation of everything retrieved. 

			curr_tag_chunk = actr_model.syntax_chunks[curr_tag]

			if print_stages:
				print('retrieved tag', curr_tag)
				# print('chunk', curr_tag_chunk)

			combined = ccg.combine(tag = curr_tag_chunk, parse_state =  goal_buffer[-1], tr_rules = tr_rules)

			if print_stages:
				print(convert_to_rule(goal_buffer[-1]),convert_to_rule(curr_tag_chunk), convert_to_rule(combined))

			goal_buffer = (i, combined)

			if combined: #i.e., not None
				# i += 1
				found_valid_tag = True
				break #stop looking for more tags
			else:
				# goal_buffer_str = (i, convert_to_rule(combined))
				# inhibition[goal_buffer_str] = inhibition.get(goal_buffer, [])
				# if len(inhibition[goal_buffer_str]) < 10: #arbitrary
				# 	inhibition[goal_buffer_str].append(curr_tag)
				poss_tags.remove(curr_tag)

				#j +=1

		if i == len(words)-1 and combined not in end_states:
			found_valid_tag = False

		if found_valid_tag:
			supertags.append(curr_tag)
			parse_states.append(combined)

			# check if there should be null
			inhibit_key = (i,convert_to_rule(combined))
			curr_inhibition = inhibition.get(inhibit_key, [])
			if words[i] in []:
				null_el = predict_null(actr_model, curr_word, curr_tag, curr_inhibition, curr_time, print_prob=True, use_priors=use_priors) 
			else:
				null_el = predict_null(actr_model, curr_word, curr_tag, curr_inhibition, curr_time, use_priors=use_priors) 
			if null_el: 
				words.insert(i+1, null_el) #next word to process should be null_el
				act_vals.insert(i+1, [])
			i+=1 #go to next word
		else:
			reanalyze_ind = get_reanalyze_ind(actr_model, supertags, words)
			# print('reanalyze_ind', reanalyze_ind)
			num_to_pop = i - reanalyze_ind
			for k in range(num_to_pop):
				removed_tag = supertags.pop()
				removed_parse_state = parse_states.pop() 


				# add inhibition to all rules along the way
				
				key = (i-1,convert_to_rule(parse_states[-1]))
				inhibition[key] = inhibition.get(key, [])
				inhibition[key].append({'tag': removed_tag, 'time': curr_time})

				# if len(supertags)> 0 and parse_states[-1] in actr_model.null_mapping: #if prev tag was 
				if words[i] in null_words: #e.g., 'comp_del'
					removed_null = words.pop(i)
					inhibition[key].append({'tag': removed_null, 'time': curr_time}) #used in predict_null

					del act_vals[i]
				else: #add inhibition to the non-null choice
					inhibition[key].append({'tag': 'not-null', 'time': curr_time}) #used in predict_null
				i -= 1
				# if len(inhibition[key]) < 7: #arbitrary
				# 	inhibition[key].append(removed_tag)
 				# if removed_tag in null_els: #define null_els
				# 	del words[i]  #because you are deleting your hypothesis about null elements too

				
				# prev_parse_state_str = convert_to_rule(parse_states[-1])

				# Add inhibition
				# key = (i-1,convert_to_rule(parse_states[-1]))
				# inhibition[key] = inhibition.get(key, [])
				

				# if len(inhibition[key]) < 10: #arbitrary
				# inhibition[key].append(removed_tag)
				## Inhibit:

				# print('inhibition', inhibition)

				
			# key = (i-1,convert_to_rule(parse_states[-1]))
			# inhibition[key] = inhibition.get(key, [])
			# inhibition[key].append({'tag': removed_tag, 'time': curr_time})
				##think do I want to do anything with the popped things??
			assert i == reanalyze_ind  #will go back to reanalysis ind in next loop

			# if i > 0: #there is no previous tag for first word
			# 	prev_tag = supertags[-1]
			# 	prev_parse_state_str = convert_to_rule(parse_states[-1])
			# 	prev_goal_buffer = (reanalyze_ind,prev_parse_state_str)

				#add inhibition

				# inhibition[prev_goal_buffer] = inhibition.get(prev_goal_buffer, [])
				# print(inhibition[prev_goal_buffer])
				# if len(inhibition[prev_goal_buffer]) < 3: #arbitrary
				# 	inhibition[prev_goal_buffer].append(prev_tag)
	
	# print(num_iters, max_iters)
	if num_iters >= max_iters:
		return(None, None, None, None)
	else:
		return(parse_states, supertags, words, act_vals)

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
	Input: list of supertags
	Output: an index of one of the supertags. The sampling is weighted by the number of other tags that could have been chosen. 
	"""
	options = {}
	for i in range(len(supertags)): #for each supertag generated (i.e., words processed)
		competitors = actr_model.lexical_chunks[words[i]]['syntax'] # number of alternative supertags
		options[i] = len(competitors)

	rand_ind = weighted_sample(options)
	return(rand_ind)
	# return(len(supertags)-1)

def predict_null(actr_model, word, curr_tag, curr_inhibition, curr_time, print_prob =False, use_priors=True):
	## Look at base level activation of null things. 
	## Compare to base level activation of all non-null things.
	## Maybe I want a kind of bigram model as well? So for every word, count of the number of times other words came after?

	## currently I am not adding inhibition to the null... 

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
		# print(curr_act_dict)


		# null_token =  actr_model.null_mapping[tag]


		# #null_token_act = sum(actr_model.lexical_act[null_token].values())
		# null_token_act = 0
		# ## Add in base activation of all possible null tags
		# for null_tag in actr_model.lexical_chunks[null_token]['syntax']:
		# 	null_token_act += actr_model.base_act[null_tag] + np.random.normal(0, actr_model.noise_sd)

		# # note: a val should never be inifite, but right now I think eos is infinite?
		# total_act = sum([val+np.random.normal(0, actr_model.noise_sd) for val in actr_model.base_act.values() if val != math.inf])


		# p_null_token = (null_token_act+eps)/(total_act + eps*len(actr_model.base_act))
		
		# if print_prob:
		# 	print(actr_model.base_act.items())
		# 	print(p_null_token, null_token_act, total_act)

		# # choice = np.random.choice([null_token, tag], p=[p_null_token, 1-p_null_token])
		# choice = np.random.choice([null_token, tag])

		# if choice != tag:
		#	return choice
		


	

def generate_supertag(actr_model, goal_buffer, curr_word, inhibition, poss_tags, curr_time, use_priors = True):
	curr_act_dict = {tag:0 for tag in poss_tags}

	if use_priors:
		# Add in base level activation
		for tag in curr_act_dict:
			curr_act_dict[tag] = actr_model.base_act[tag]

	    # Add in activation from the word
		for tag in curr_act_dict:
			curr_act_dict[tag] += actr_model.lexical_act[curr_word][tag]  

	
	# Add in inhibition from the goal_buffer for things that did not work
	goal_buffer_str = (goal_buffer[0], convert_to_rule(goal_buffer[1]))
	if goal_buffer_str in inhibition:
		curr_inhibition = inhibition[goal_buffer_str]
		avg_inhibition = actr_model.max_activation/len(curr_inhibition)
		# maybe instead of avg_inhibition, I can just let this decay over time (just like activation?)

		for tag in curr_act_dict:
			tag_inhibition = actr_model.compute_inhibition(tag, curr_inhibition, curr_time)
			curr_act_dict[tag] -= tag_inhibition
			# curr_act_dict[tag] -= tag_inhibition*actr_model.max_activation  #why was I multiplying??
			# tag_freq = curr_inhibition.count(tag)
			# curr_act_dict[tag] -= tag_freq*avg_inhibition

		# print('after inhibition', curr_act_dict)
    
    # Add in random noise

	for tag in curr_act_dict:  
		curr_act_dict[tag] += np.random.normal(0, actr_model.noise_sd)

	supertag = max(curr_act_dict, key=lambda key:curr_act_dict[key])
	act_val = curr_act_dict[supertag]


	return(supertag, act_val)


# def compute_inhibition(tag, inhibition_list, curr_time, decay):
# 	inhibition_val = 0
# 	for item in inhibition_list:
# 		if item['tag'] == tag:
# 			time_since_item = curr_time + actr_model.eps - item['time']
# 			inhibition_val += np.power(time_since_item, decay)

# 	if inhibition_val == 0:
# 		return 0
# 	else:
# 		return np.log(inhibition_val)
	# inhibition = 

	# return(max(0,inhibition))


## TO FIGURE OUT: Why do I get inf values

# def supertag_sentence(actr_model, sentence, print_stages=False, end_states=[{'left': 'end', 'right': '', 'combinator': ''}]):
# 	"""
# 	Inputs:
# 		actr_model: only the declartive memory and type raising rules will differ across theories
# 		sentence: string in lowercase. words separated by space.
# 		print_stages: Boolean, will print all stages of supertagging if True
# 		end_states: Valid end states. For full sentences this will always be 'end' but can differ for partial prompts. 
	
# 	Outputs:
# 		goal_buffer: the final parse state (should be one of the end_states)
# 		supertags: the sequence of tags for words (one per word)
# 		words: words in the sentence + any additional null elements
# 		act_vals: the activation value for all the supertags when they were retrieved.
# 	"""
# 	words = sentence.split() 
# 	tr_rules = actr_model.type_raising_rules

# 	#initialize variables
# 	supertags = ['' for word in words]
# 	act_vals = [[] for word in words] #activation values for all tags considered.
# 	goal_states = [None] 
# 	inhibition = {}
# 	i = 0

# 	while(i < len(words)):
# 		goal_buffer = goal_states[-1] #i.e. current parse state
# 		curr_word = words[i]
# 		poss_tags = actr_model.lexical_chunks[curr_word]['syntax'].copy()
# 		poss_tags = remove_impossible(poss_tags) #need to implement

# 		if(print_stages):
# 			print('=======================')
# 			print(curr_word)
# 			print(goal_buffer)

# 		j = 0
# 		while (j < len(poss_tags)): #while there are possible tags
# 			curr_tag, curr_act = generate_supertag(actr_model, goal_buffer, curr_word, excluded_tags, poss_tags)

# 			curr_tag_chunk = actr_model.syntax_chunks[curr_tag]

# 			# try to combine with goal state
# 			combined = combine(tag = curr_tag_chunk, goal_buffer =  goal_buffer, tr_rules = type_raising_rules)

# 			if combined != None:
# 				if curr_word == words[-1]:
# 					if combined not in end_states: #last word is not a valid end state
# 						combined = False 
# 					else:
# 						combined = True
# 				else:
# 					goal_buffer = combined #update parse state
# 					supertags[i] = curr_tag #associate word with supertag

#                 	goal_states.append(goal_buffer) 

#                 	if curr_tag == 'NP_CP': #more processing if CP
#                 		cp_type, cp_act = sample_cp2(actr_model, curr_bad_cp[i-1]) ## Note think about this curr_bad_cp and how it interfaces with reanalysis

#                 		if cp_type != 'NP_CP': #if NP_CP nothing else to do
#                 			i+=1 #we want to increment i because we are adding a new element
#                 			## WRITE A FUNCTION HERE THAT HANDLES THE DIFFERENT TYPES OF GRAMMARS (i.e. WD vs WD2) INSTEAD OF HAVING TWO SUPERTAG FUNCTIONS.
#                 			words.insert(i, 'comp_del') #i+1 because you want to insert after the current word. 
#                         	supertags.insert(i, cp_type)
#                         	act_vals.insert(i, [cp_act])

#                 			new_state = combine(tag = comp_chunk, goal_buffer = goal_buffer, tr_rules = type_raising_rules)

#                 			goal_states.append(new_state)
#                 break # as long as combined was not None, break. 
#         	## There was an else with excluded tag. Not sure why it is important
#         if combined != None: #there was a valid parse
#         	i += 1 #move on to the next word
#         else:
#         	i -= 1 # go to previous word
#         	reanalyze() # to implement. Might not need to if remove_impossible handles it correctly. 
#         	## Needs to also take into account the null elements 

#         	## There were also a bunch of corner cases for V_intrans and the last two tags being null elements. I think this should not be necessary after incorporating the end state, but maybe I am wrong. 	

#     return(goal_buffer, supertags, words, act_vals)


# def train(actr_model, model_type, sents):
# 	num_prev_tags = sum(actr_model.base_count.values())

# 	if model_type == 'ep':
# 		supertag_sentence = supertag_sentence_ep
# 	else:
# 		supertag_sentence = supertag_sentence_wd

# 	for i,sent in enumerate(sents):
# 		# curr_tag_list = tags[i].split()

# 		final_state, tags, words = supertag_sentence_wd(actr_model, sents)


# 		ccg_tag_list = [(words[n], tags[n]) for n in range(len(word_list))]

# 		for j, pair in enumerate(ccg_tag_list):
# 			num_prev_tags += 1
# 			word = pair[0]
# 			tag = pair[1]

# 			actr_model.lexical_count[word][tag] +=1

# 			actr_model.base_count[tag] += 1

# 			actr_model.base_instance[tag].append(num_prev_tags)





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


if __name__ == '__main__':
	from model import actr_model
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


	# tags = ['Det', 'Adj', 'NP_VoiceP', 'Vt_pass', 'Prep', 'Det', 'NP', 'aux', 'Adj', 'eos']
	# # tags = []
	# curr_parse = None 
	# for tag in tags:
	# 	tag_chunk = syntax_chunks_ep[tag]
	# 	print(convert_to_rule(curr_parse), '+', convert_to_rule(tag_chunk))
	# 	curr_parse = ccg.combine(curr_parse, tag_chunk, type_raising_rules)

	
	# 	print('end',convert_to_rule(curr_parse))
	# 	print()

	global_sd = 1

	# Hyperparameters that do not change
	decay = 0.5           # Standard value for ACT-R models
	max_activation = 1.5  # Standard value for ACT-R models
	latency_exponent = 1  # Will always set this to 1 (because then it matches eqn 4 from Lewis and Vasisth)
	noise_sd = np.random.uniform(0.2,0.5)
	latency_factor = np.random.beta(2,6)
	max_iters = 1000

	actr_model_wd = actr_model(decay,
							  max_activation,
							  noise_sd,
							  latency_factor,
							  latency_exponent,
							  syntax_chunks_wd,
							  lexical_chunks_wd,
							  type_raising_rules,
							  null_mapping,
							  max_iters)

	actr_model_ep = actr_model(decay,
							  max_activation,
							  noise_sd,
							  latency_factor,
							  latency_exponent,
							  syntax_chunks_ep,
							  lexical_chunks_ep,
							  type_raising_rules,
							  null_mapping,
							  max_iters)

	print(actr_model_wd.lexical_null_act['defendant'])

	partial_states = [{'left': 'DP', 'right': 'PP', 'combinator': '/'}, {'left': 'TP', 'right': 'DP', 'combinator': '/'}, {'left': 'end', 'right': '', 'combinator': ''}]
	# with open('test_sents_supertagger.txt') as f:
	with open('./data/train10000.txt') as f:
		for line in f:
			print(line)
			parse_states, supertags, words, act_vals = supertag_sentence(actr_model_wd, line, end_states=partial_states, print_stages=False)
			while parse_states == None:
				parse_states, supertags, words, act_vals = supertag_sentence(actr_model_wd, line, end_states=partial_states, print_stages=False)
			# print('WD', supertags)
			# print(parse_states)
			# print(words)
			parse_states, supertags, words, act_vals = supertag_sentence(actr_model_ep, line, end_states=partial_states, print_stages=False)
			while parse_states == None:
				parse_states, supertags, words, act_vals = supertag_sentence(actr_model_ep, line, end_states=partial_states, print_stages=False)
			# print('EP', supertags)
			# print()



## TO DO: Write a bunch of tests

# word_list = ['the', 'defendant', 'comp_del', 'aux_del', 'examined']
# tag_list = ['Det', 'NP_CP', 'compsubj_null', 'aux_pass', 'Vt_pass']
# print_states(type_raising_rules, tag_list, word_list)

# word_list = 'its  engineer talked-about  an  politician and  participated playfully to their good home .'.split()

# tag_list = ['Det', 'NP', 'Vt_act', 'Det', 'NP', 'conj', 'Vt_loc', 'Adv', 'Prep', 'Det', 'Adj', 'NP', 'eos']

# word_list = 'the lawyer who was examined by the defendant loved the unreliable cat .'.split()
# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'Adj', 'NP', 'eos']

# print_states(type_raising_rules, tag_list, word_list)
# tag = {'left': '(TP\\DP)', 'right': 'DP', 'combinator': '/'}

# chunk = {'left': ' (TP/(TP\\DP))', 'right': '', 'combinator': ''}



# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP', '.']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat', 'eos']


# chunk = {'left': 'TP', 'right': '(TP\\DP)', 'combinator': '/'}

# print(combine(tag, chunk))

# x/y x 

# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat', '.']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')


# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_loc', 'Prep', 'Det', 'Adj', 'NP', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'arrived', 'at', 'the', 'unreliable', 'palace', '.']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')

# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'aux', 'Adj', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'was', 'unreliable', '.']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')



# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'V_intrans', 'Adv', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'sang', 'beautifully', '.']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')


# tag_list = ['Det', 'NP_VoiceP', 'Vt_pass', 'Prep', 'Det', 'NP', 'aux', 'Adj', 'eos']

# word_list = ['the', 'defendant', 'examined', 'by', 'the', 'lawyer', 'was', 'unreliable', '.']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')


# tag_list = ['Det', 'NP_CP_null', 'compsubj_null', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')

# tag_list = ['Det', 'NP_VoiceP', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# word_list = ['the', 'defendant', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')

# word_list = ['the', 'defendant', 'who', 'was', 'being','examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_prog', 'ProgP', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')


# word_list = ['the', 'defendant', 'being','examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_ProgP', 'ProgP', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')


# word_list = ['the', 'defendant', 'who', 'examined','the', 'lawyer', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP', 'compsubj', 'Vt_act',  'Det', 'NP', 'Vt_act', 'Det', 'NP']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')

# word_list = ['the', 'defendant', 'who','the', 'lawyer', 'examined', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP', 'compobj', 'Det', 'NP', 'Vt_act', 'Vt_act', 'Det', 'NP']

# print_states(type_raising_rules, tag_list, word_list)



# word_list = ['the', 'defendant', 'who_del', 'the', 'lawyer', 'examined', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP_null', 'compobj_null', 'Det', 'NP', 'Vt_act', 'Vt_act', 'Det', 'NP']


# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP', 'conj', 'V_intrans', 'Adv', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat', 'and',  'sang', 'unreliably', '.']

# print_states(type_raising_rules, tag_list, word_list)

# print('-----------------------------')

## This is wrong parse


# tag_list =['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP']


# word_list = ['the', 'defendant', 'who', 'was', 'loved', 'by', 'his', 'tenants']



# print_states(type_raising_rules, tag_list, word_list)


# print(add_parens('((TP\\DP)/DP)/NP'))

# print(add_parens('((TP\\DP)/DP)/NP)))'))

# print(add_parens('(((TP\\DP)/DP)/NP)'))
