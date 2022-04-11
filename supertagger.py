import re
# import declmem_v2
import math
import random
from copy import deepcopy
import numpy as np
import pickle

## To do: create a model object that stores the base level activation etc. 

def convert_to_rule(state):
	if state != None:
		return(state['left'] + state['combinator'] + state['right'])


def supertag_sentence_ep(actr_model, sentence, print_stages=False):
	# print(sentence)
	end_state = {'left': 'end', 'right': '', 'combinator': ''}
	words = sentence.split()
	supertags = ['' for word in words]
	act_vals = [[] for word in words] #activation values for all tags considered. 
	goal_states = [None]

	# curr_bad_tags = []
	curr_bad_tags = [[] for word in words]

	# curr_bad_cp = set()
	# curr_bad_aux = set()
	# curr_bad_tag = ''
	cp_type = ''
	aux_type = ''

	i = 0
	while(i < len(words)):
		
		goal_buffer = goal_states[-1]
		curr_word = words[i]
		if(print_stages):
			print('=======================')
			print(curr_word)
			print(supertags)


		poss_tags = actr_model.lexical_chunks[curr_word]['syntax']
		poss_tags = [x for x in poss_tags if x not in curr_bad_tags[i]]
		# poss_tags = [x for x in poss_tags if x != curr_bad_tag]

		# print(curr_word, poss_tags)

		j = 0
		excluded_tags = []
		combined = False


		if len(poss_tags) == 0:
			# curr_bad_tag = ''
			# curr_bad_tags = []
			curr_bad_tags[i] = []
			
		# print(curr_word)
		# print('len', len(poss_tags))
		# if curr_word == 'was':
		# 	print('was poss tags', poss_tags)
		# 	print('was bad tags', curr_bad_tags)
		

		while(j < len(poss_tags)):
			# print('j',j)
			curr_tag, curr_act = generate_supertag(actr_model, goal_buffer, curr_word, excluded_tags, poss_tags)   #after applying any possible type raising

			curr_tag_chunk = actr_model.syntax_chunks[curr_tag]

			act_vals[i].append(curr_act)

			# print('curr_tag', curr_tag)
			# print('curr_tag_chunk', curr_tag_chunk)
			# print('goal_buffer', goal_buffer)

			combined = combine(curr_tag_chunk, goal_buffer)
			
			if combined != None:  #i.e. there is a combined state

				if curr_word == '.' and combined != end_state:
					combined=False
					# print('REACHED HERE')
					# print(supertags)
					break

				goal_buffer = combined
				
				supertags[i] = curr_tag

				goal_states.append(goal_buffer)

				i +=1

				if goal_buffer['right'] == 'CP_null':
					comp_chunk = actr_model.syntax_chunks['compobj_null']
					# print('words before', words)
					words.insert(i, 'comp_del')
					# print('words after', words)
					supertags.insert(i, 'compobj_null')
					act_vals.insert(i, [actr_model.max_activation]) #only one so max activation ? REVISIT THIS DECISION. 

					new_state = combine(comp_chunk, goal_buffer)
					curr_bad_tags.insert(i, [])
					goal_states.append(new_state)
					# curr_bad_tags = []
					
					i+=1

				

				break

			else:
				excluded_tags.append(curr_tag)
				j+=1
				# NOTE: this is not taking into account time for reanalysis. 

		# re-analysis while removing the excluding the previous tag
		# Not sure if this is the best reanalysis strategy!
		if not combined:
			curr_bad_tags[i] = [] # refresh the memory of the current word that could not be processed
			i -=1
			goal_states.pop(-1)

			if words[i] == 'comp_del':
				# Remove comp del
				
				goal_states.pop(-1) # remove aux
				words.pop(i)
				supertags.pop(i)
				curr_bad_tags.pop(i)

				# add activation to previous word
				x = act_vals.pop(i) 
				act_vals[i-1].extend(x) 


				curr_bad_tags[i] = []

				i-=1

			curr_bad_tags[i].append(supertags[i])
			supertags[i] = ''

	# manually change the last NP tag to NP. 
	# to do: make sure to get this with EOS somehow
	if supertags[-2] in ['compobj_null']:
		supertags.pop(-2)
		words.pop(-2)
		act_vals.pop(-2)  #this is algorithm error so no need to add

	if supertags[-2] in ['NP_CP', 'NP_CP_null', 'NP_VoiceP', 'NP_ProgP']:
		supertags[-2] = 'NP'

	if 'V_intrans' in supertags:
		intrans_ind	= supertags.index('V_intrans')

		# if supertags[intrans_ind-1] in ['aux_pass', 'aux_prog']:
		# 	supertags.pop(intrans_ind-1)

		# if supertags[intrans_ind-1] in ['compobj_null', 'compsubj_null']:
		# 	supertags.pop(intrans_ind-1)

		if supertags[intrans_ind-1]  in ['NP_CP', 'NP_CP_null', 'NP_VoiceP', 'NP_ProgP']:
			supertags[intrans_ind-1] = 'NP'

		# print(supertags, words)

	return(goal_buffer, supertags, words, act_vals)


def supertag_sentence_wd(actr_model, sentence, print_stages=False):
	end_state = {'left': 'end', 'right': '', 'combinator': ''}
	# print(sentence)
	words = sentence.split()
	supertags = ['' for word in words]
	act_vals = [[] for word in words] #activation values for all tags considered.
	goal_states = [None]

	# curr_bad_tags = []
	curr_bad_tags = [set() for word in words]
	curr_bad_cp = [set() for word in words]
	curr_bad_aux = [set() for word in words]
	# curr_bad_cp = set()
	# curr_bad_aux = set()
	# curr_bad_tag = ''
	

	# print(curr_bad_tags)

	i = 0
	while(i < len(words)):
		# cp_type = ''
		# aux_type = ''
		# print('=======================')
		# print(i)
		goal_buffer = goal_states[-1]
		curr_word = words[i]

		if(print_stages):
			print('=======================')
			print(curr_word)
			print(goal_buffer)

		# print(curr_word)
		# print('goal_states', goal_states)
		# print('goal_buffer', goal_buffer)
		# for j, supertag in enumerate(supertags):
		# 	print(words[j], supertag, curr_bad_tags[j], curr_bad_cp[j])
		# print('supertags', supertags)
		# print('words', words)
		# print('words', words)


		poss_tags = actr_model.lexical_chunks[curr_word]['syntax']
		poss_tags = [x for x in poss_tags if x not in curr_bad_tags[i]]

		# print('curr_bad_tags', curr_bad_tags[i])
		# print('poss_tags', poss_tags)
		# print('curr_bad_cp', curr_bad_cp[i])
		# print('curr_bad_aux', curr_bad_aux[i])


		# print('curr word', curr_word)
		# print('goal_buffer', goal_buffer)

		j = 0
		excluded_tags = []
		combined = False

		if len(poss_tags) == 0:
			# print('Getting re-set here. Word is', curr_word)
			curr_bad_tags[i] = set() 
			curr_bad_cp[i] = set()
			curr_bad_aux[i] = set()
			# curr_bad_cp = set()
			# curr_bad_aux = set()

		# print('bad tag for word', curr_word, curr_bad_tags[i])

		while(j < len(poss_tags)):
			curr_tag, curr_act = generate_supertag(actr_model, goal_buffer, curr_word, excluded_tags, poss_tags)   #after applying any possible type raising
			

			curr_tag_chunk = actr_model.syntax_chunks[curr_tag]

			combined = combine(curr_tag_chunk, goal_buffer)

			if(print_stages):
				print('considering tag:', curr_tag)
				print('combined state:', combined)

			act_vals[i].append(curr_act)
			
			if combined != None:  #i.e. there is a combined state

				if curr_word == '.' and combined != end_state:
					combined=False
					# print('REACHED HERE')
					# print(supertags)
					break

				goal_buffer = combined
				
				supertags[i] = curr_tag

				goal_states.append(goal_buffer)
				# curr_bad_tag = ''
				# curr_bad_tags[i] = []
				i +=1

				if goal_buffer['right'] == 'CP_null':
					# print('supertags when CP_null', supertags)
					# print('curr_bad_cp when CP_null', curr_bad_cp)
					# print('words at i-1', words[i-1])


					# i-1 because the CP is associated with the noun
					cp_type, cp_act = sample_cp(actr_model, curr_bad_cp[i-1])
					# print('cp_type', cp_type)
					# print('curr_bad_cp', curr_bad_cp[i])
					if cp_type != None:
						comp_chunk = actr_model.syntax_chunks[cp_type]

					if cp_type == 'compobj_null':
						words.insert(i, 'comp_del')
						supertags.insert(i, cp_type)
						act_vals.insert(i, [cp_act])
						new_state = combine(comp_chunk, goal_buffer)
						curr_bad_tags.insert(i, [])
						curr_bad_cp.insert(i, set())
						curr_bad_aux.insert(i, set())
						goal_states.append(new_state)
						# curr_bad_tags = []
						
						i+=1

					elif cp_type == 'compsubj_null':
						words.insert(i, 'comp_del')
						supertags.insert(i, cp_type)
						act_vals.insert(i, [cp_act])
						curr_bad_tags.insert(i, [])
						curr_bad_cp.insert(i, set())
						curr_bad_aux.insert(i, set())
						new_state = combine(comp_chunk, goal_buffer)

						goal_states.append(new_state)
						i+=1
						# i-2 because aux is associated with the noun
						aux_type, aux_act = sample_aux(actr_model, curr_bad_aux[i-2])

						if aux_type != None:

							aux_chunk = actr_model.syntax_chunks[aux_type]

							words.insert(i, 'aux_del')
							supertags.insert(i, aux_type)
							act_vals.insert(i, [aux_act])
							curr_bad_tags.insert(i, [])
							curr_bad_cp.insert(i, set())
							curr_bad_aux.insert(i, set())
							new_state = combine(aux_chunk, goal_states[-1])

							goal_states.append(new_state)
							i+=1

							# curr_bad_tags = []

				# print('combined state:', goal_states[-1])
				break

			else:
				excluded_tags.append(curr_tag)
				j+=1


		# re-analysis while removing the excluding the previous tag
		# Not sure if this is the best reanalysis strategy!
		if not combined:
			# print('Reached here also')
			curr_bad_tags[i] = set() 
			i -=1
			# print('goal_states0', goal_states)
			goal_states.pop(-1)
			# print('goal_states1', goal_states)
			has_aux_del = False

			# curr_bad_tag = supertags[i]
			if words[i] == 'aux_del':
				has_aux_del = True
				# curr_bad_aux.add(aux_type)
				# print('goal_states1', goal_states)
				goal_states.pop(-1) # remove aux_del
				words.pop(i)
				bad_aux_tag = supertags.pop(i)
				curr_bad_tags.pop(i)
				curr_bad_cp.pop(i)
				curr_bad_aux.pop(i)

				x = act_vals.pop(i)
				act_vals[i-1].extend(x)  # should this be i-2 ? 

				curr_bad_tags[i] = set() 


				#curr_bad_aux[i-2].add(aux_type)  #you want it to be for the prev noun 
				curr_bad_aux[i-2].add(bad_aux_tag)

				aux_type = ''

				# print('curr_bad_aux 2', curr_bad_aux[i-2])
				# if len(curr_bad_aux) == 2:
				if len(curr_bad_aux[i-2]) == 2:
					# curr_bad_cp.add(cp_type)   #tried all the aux, so CP doesn't work. 
					#curr_bad_cp[i-2].add(cp_type)
					curr_bad_cp[i-2].add(supertags[i-1]) #cp would be the prev tag
					cp_type = ''
		
				i -=1



			if words[i] == 'comp_del':
				# print('hellooo')
				# print('has_aux_del', has_aux_del)
				# Remove comp del
				
				# print('goal_states2', goal_states)
				goal_states.pop(-1) # remove aux
				# print('goal_states3', goal_states)
				words.pop(i)
				supertags.pop(i)
				curr_bad_tags.pop(i)
				curr_bad_cp.pop(i)
				curr_bad_aux.pop(i)    #this is getting 

				x = act_vals.pop(i)
				act_vals[i-1].extend(x)


				curr_bad_tags[i] = set() 

				i-=1

				if not has_aux_del: #there was no aux but there is comp_del
					# curr_bad_cp.add(cp_type)
					# print('REACHED HERE')
					curr_bad_cp[i].add(cp_type)
					cp_type = ''

					# print(curr_bad_cp[i], words[i])

				

				
				# if len(curr_bad_cp) == 2:
				# 	curr_bad_tags[i].append(supertags[i])

				if len(curr_bad_cp[i]) == 2:
					# print('REACHED HERE ALSO')
					curr_bad_tags[i].add(supertags[i])
					# print(curr_bad_tags[i], words[i])

				supertags[i] = ''

			else:
				curr_bad_tags[i].add(supertags[i])
				supertags[i] = ''
		if(print_stages):
			print(supertags)
		# print('supertags after',supertags)
			# print(curr_bad_tags)
		# print('--------')
		# for j, supertag in enumerate(supertags):
		# 	print(words[j], supertag, curr_bad_tags[j], curr_bad_cp[j])

	# manually change the last NP tag to NP. 
	# to do: make sure to get this with EOS somehow
	if supertags[-2] in ['aux_pass', 'aux_prog']:
		supertags.pop(-2)
		words.pop(-2)
		act_vals.pop(-2)

	if supertags[-2] in ['compobj_null', 'compsubj_null']:
		supertags.pop(-2)
		words.pop(-2)
		act_vals.pop(-2)

	if supertags[-2]  in ['NP_CP', 'NP_CP_null']:
		supertags[-2] = 'NP'


	# if words[-2] =='aux_del':
	# 	words.pop(-2)

	# if words[-2] == 'comp_del':
	# 	words.pop(-2)

	# manually ensure that intransitive sentences have correct NP
	# to do: have a more restrictive intrans label that solves this
	if 'V_intrans' in supertags:
		
		intrans_ind	= supertags.index('V_intrans')

		if supertags[intrans_ind-1] in ['aux_pass', 'aux_prog']:
			supertags.pop(intrans_ind-1)
			words.pop(intrans_ind-1)
			act_vals.pop(intrans_ind-1)
			intrans_ind-=1

		if supertags[intrans_ind-1] in ['compobj_null', 'compsubj_null']:
			supertags.pop(intrans_ind-1)
			words.pop(intrans_ind-1)
			act_vals.pop(intrans_ind-1)
			intrans_ind-=1

		if supertags[intrans_ind-1]  in ['NP_CP', 'NP_CP_null']:
			supertags[intrans_ind-1] = 'NP'

	## manually ensure that conj sentences have the correct NP type


	return(goal_buffer, supertags, words, act_vals)

# def supertag_sentence_wd(actr_model, sentence):
# 	words = sentence.split()
# 	supertags = ['' for word in words]
# 	goal_states = [None]

# 	curr_bad_tags = []
# 	curr_bad_cp = set()
# 	curr_bad_aux = set()
# 	# curr_bad_tag = ''
# 	cp_type = ''
# 	aux_type = ''

# 	i = 0
# 	while(i < len(words)):
# 		# print('=======================')
# 		goal_buffer = goal_states[-1]
# 		curr_word = words[i]


# 		poss_tags = actr_model.lexical_chunks[curr_word]['syntax']
# 		poss_tags = [x for x in poss_tags if x not in curr_bad_tags]

# 		# print('curr word', curr_word)
# 		# print('goal_buffer', goal_buffer)

# 		j = 0
# 		excluded_tags = []
# 		combined = False

# 		if len(poss_tags) == 0:
# 			curr_bad_tags = []
# 			curr_bad_cp = set()
# 			curr_bad_aux = set()


# 		while(j < len(poss_tags)):
# 			curr_tag = generate_supertag(actr_model, goal_buffer, curr_word, excluded_tags, poss_tags)   #after applying any possible type raising

# 			# print('considering tag:', curr_tag)

# 			curr_tag_chunk = actr_model.syntax_chunks[curr_tag]

# 			combined = combine(curr_tag_chunk, goal_buffer)
			
# 			if combined != None:  #i.e. there is a combined state
# 				goal_buffer = combined
				
# 				supertags[i] = curr_tag

# 				goal_states.append(goal_buffer)
# 				# curr_bad_tag = ''
# 				# curr_bad_tags = []
# 				i +=1

# 				if goal_buffer['right'] == 'CP_null':
# 					cp_type = sample_cp(actr_model, curr_bad_cp)
# 					comp_chunk = actr_model.syntax_chunks[cp_type]
# 					if cp_type == 'compobj_null':
# 						words.insert(i, 'comp_del')
# 						supertags.insert(i, cp_type)
# 						new_state = combine(comp_chunk, goal_buffer)
# 						goal_states.append(new_state)
# 						curr_bad_tags = []
						
# 						i+=1

# 					elif cp_type == 'compsubj_null':
# 						words.insert(i, 'comp_del')
# 						supertags.insert(i, cp_type)
# 						new_state = combine(comp_chunk, goal_buffer)

# 						goal_states.append(new_state)
# 						i+=1

# 						aux_type = sample_aux(actr_model, curr_bad_aux)

# 						if aux_type != None:

# 							aux_chunk = actr_model.syntax_chunks[aux_type]

# 							words.insert(i, 'aux_del')
# 							supertags.insert(i, aux_type)
# 							new_state = combine(aux_chunk, goal_states[-1])

# 							goal_states.append(new_state)
# 							i+=1

# 							curr_bad_tags = []

# 				# print('combined state:', goal_states[-1])
# 				break

# 			else:
# 				excluded_tags.append(curr_tag)
# 				j+=1

# 		# re-analysis while removing the excluding the previous tag
# 		# Not sure if this is the best reanalysis strategy!
# 		if not combined:
# 			i -=1
# 			goal_states.pop(-1)

# 			# curr_bad_tag = supertags[i]
# 			if words[i] == 'aux_del':
# 				curr_bad_aux.add(aux_type)

# 				if len(curr_bad_aux) == 2:
# 					curr_bad_cp.add(cp_type)   #tried all the aux, so CP doesn't work. 
# 				goal_states.pop(-1) # remove aux
# 				words.pop(i)
# 				supertags.pop(i)
# 				i -=1

# 			if words[i] == 'comp_del':
# 				# Remove comp del
				
# 				goal_states.pop(-1) # remove aux
# 				words.pop(i)
# 				supertags.pop(i)

# 				i-=1

# 				supertags[i] = ''

# 				if len(curr_bad_cp) == 2:
# 					curr_bad_tags.append(supertags[i])

# 				# if len(curr_bad_aux)


# 			else:
# 				curr_bad_tags.append(supertags[i])
# 				supertags[i] = ''


 

# 	return(goal_buffer, supertags)



# def sample_cp(actr_model, curr_bad_cp):
# 	if len(curr_bad_cp) < 3:
# 		comp = ''
# 		while comp == '':
# 			comp_types = ['compobj_null', 'compsubj_null']
# 			comp_types = [x for x in comp_types if x not in curr_bad_cp]

# 			if len(comp_types) > 0:

# 				comp_act_dict = {key:actr_model.base_act[key] for key in comp_types}
# 				# consider adding lexical activation from nouns ??

# 				if sum(comp_act_dict.values()) == 0:  
# 					comp = np.random.choice(comp_types)
# 				else:
# 					probs = np.array(list(comp_act_dict.values()))/sum(comp_act_dict.values())
# 					comp = np.random.choice(comp_types, p=probs)

# 				# if comp not in curr_bad_cp:
# 				return(comp)

# def sample_cp(actr_model, curr_bad_cp):

# 	comp_types = ['compobj_null', 'compsubj_null']
# 	comp_types = [x for x in comp_types if x not in curr_bad_cp]
# 	print('curr_bad_cp', curr_bad_cp)
# 	print('possible comp_types', comp_types)

# 	if len(comp_types) > 0:

# 		comp_act_dict = {key:actr_model.base_act[key] for key in comp_types}
# 		# consider adding lexical activation from nouns ??

# 		if sum(comp_act_dict.values()) == 0:  
# 			comp = np.random.choice(comp_types)
# 		else:
# 			probs = np.array(list(comp_act_dict.values()))/sum(comp_act_dict.values())
# 			comp = np.random.choice(comp_types, p=probs)

# 		# if comp not in curr_bad_cp:
# 		return(comp)

def sample_cp(actr_model, curr_bad_cp):

	comp_types = ['compobj_null', 'compsubj_null']
	comp_types = [x for x in comp_types if x in actr_model.syntax_chunks]  #compsubj_null not in ep
	comp_types = [x for x in comp_types if x not in curr_bad_cp]

	if len(comp_types) > 0:

		comp_act_dict = {key:actr_model.base_act[key] for key in comp_types}
		# consider adding lexical activation from nouns ??

		if sum(comp_act_dict.values()) == 0:  
			comp = np.random.choice(comp_types)
			comp_act = actr_model.max_activation/len(comp_types)
		else:
			comp = max(comp_act_dict, key=lambda key:comp_act_dict[key])
			comp_act = comp_act_dict[comp]
			# probs = np.array(list(comp_act_dict.values()))/sum(comp_act_dict.values())
			# comp = np.random.choice(comp_types, p=probs)

		# if comp not in curr_bad_cp:
		return(comp, comp_act)
	else:
		return(None,None)


def sample_aux(actr_model, curr_bad_aux):
	aux_types = ['aux_pass', 'aux_prog']
	aux_types = [x for x in aux_types if x not in curr_bad_aux]

	aux_act_dict = {key:actr_model.base_act[key] for key in aux_types}
	# consider adding lexical activation from nouns ??

	if len(aux_types) > 0:
		if sum(aux_act_dict.values()) == 0:  
			aux = np.random.choice(aux_types)
			aux_act = actr_model.max_activation/len(aux_types)
		else:
			aux = max(aux_act_dict, key=lambda key:aux_act_dict[key])
			aux_act = aux_act_dict[aux]
			# probs = np.array(list(aux_act_dict.values()))/sum(aux_act_dict.values())
			# aux = np.random.choice(aux_types, p=probs)

		# if aux not in curr_bad_aux:
		return(aux, aux_act)

# def sample_aux(actr_model, curr_bad_aux):
# 	if len(curr_bad_aux) < 3:
# 		aux = ''
# 		while aux == '':
# 			aux_types = ['aux_pass', 'aux_prog']
# 			comp_types = [x for x in aux_types if x not in curr_bad_aux]

# 			aux_act_dict = {key:actr_model.base_act[key] for key in aux_types}
# 			# consider adding lexical activation from nouns ??


# 			if sum(aux_act_dict.values()) == 0:  
# 				aux = np.random.choice(aux_types)
# 			else:
# 				probs = np.array(list(aux_act_dict.values()))/sum(aux_act_dict.values())
# 				aux = np.random.choice(aux_types, p=probs)

# 			if aux not in curr_bad_aux:
# 				return(aux)

		
def generate_supertag(actr_model, goal_buffer, word, excluded_tags, poss_tags):
	# poss_tags = actr_model.lexical_chunks[word]['syntax']
	poss_tags = [x for x in poss_tags if x not in excluded_tags]

	curr_act_dict = {tag:0 for tag in poss_tags}

	# Add in base level activation
	for tag in curr_act_dict:
		curr_act_dict[tag] = actr_model.base_act[tag]

	# Add in activation from the word
	for tag in curr_act_dict:
		curr_act_dict[tag] += actr_model.lexical_act[word][tag]  
		# To do: remember to compute this before hand (compute prob and multiply by max_activation)

	# Add in activation from goal buffer

	#for chunk in goal_buffer:
	valid_tags = []
	for tag_label in curr_act_dict:
		tag = actr_model.syntax_chunks[tag_label]
		combined = combine(tag, goal_buffer) # if there is only one thing in the buffer, this is the same as calling combine(tag, buffer)
		if combined != None: #i.e. if it can combine
			valid_tags.append(tag_label)

	for tag_label in valid_tags:  # add equal activation to all tags? 
		#alternative: can learn association between goal state and tag. 
		tag = actr_model.syntax_chunks[tag_label]
		curr_act_dict[tag_label] += actr_model.max_activation/len(valid_tags)

	for tag in curr_act_dict:  #add in random noise
		curr_act_dict[tag] += np.random.normal(0, actr_model.noise_sd)

	supertag = max(curr_act_dict, key=lambda key:curr_act_dict[key])
	act_val = curr_act_dict[supertag]

	# all_tags = list(curr_act_dict.keys())

	# if sum(curr_act_dict.values()) == 0:  
	# 	supertag = np.random.choice(all_tags)
	# else:
	# 	probs = np.array(list(curr_act_dict.values()))/sum(curr_act_dict.values())
	# 	supertag = np.random.choice(all_tags, p=probs)


	#all_tag_probs = np.array(list(curr_act_dict.values()))/sum(curr_act_dict.values())

	#supertag = np.random.choice(all_tags, p=all_tag_probs)

	return(supertag, act_val)



def forward_appl(chunk1, chunk2):
	# print('In forward')
	# print('chunk1', chunk1)
	# print('chunk2', chunk2)
	# print(chunk1['combinator'])
	if chunk1['combinator'] == '/' and chunk2['right'] == '':
		if chunk1['right'] == chunk2['left']:
			state = {
				'left': chunk1['left'],
				'right': '',
				'combinator': ''
			}
			return(state)
	else:
		chunk2_rule = '(' + chunk2['left'] + chunk2['combinator'] + chunk2['right'] + ')'

		if chunk1['combinator'] == '/' and chunk1['right'] == chunk2_rule:
			state = {
				'left': chunk1['left'],
				'right': '',
				'combinator': ''
			}
			return(state)


def backward_appl(chunk1, chunk2):
	if chunk2['combinator'] == '\\' and chunk1['right'] == '':
		if chunk1['left'] == chunk2['right']:
			state = {
				'left': chunk2['left'],
				'right': '',
				'combinator': ''
			}

			return(state)
	else:
		# print('Reached here')
		chunk1_rule = '(' + chunk1['left'] + chunk1['combinator'] + chunk1['right'] + ')'
		# print(chunk1_rule)

		if chunk2['combinator'] == '\\' and chunk1_rule == chunk2['right']:
			state = {
				'left': chunk2['left'],
				'right': '',
				'combinator': ''
			}

			return(state)

def forward_harmonic(chunk1, chunk2):
	if chunk1['combinator'] == '/' and chunk2['combinator'] == '/':
		if chunk1['right'] == chunk2['left']:
			state = {
					'left': chunk1['left'],
					'right': chunk2['right'],
					'combinator': '/'
				}
			return(state)


def backward_harmonic(chunk1, chunk2):
	if chunk1['combinator'] == '\\' and chunk2['combinator'] == '\\':
		if chunk1['left'] == chunk2['right']:
			state = {
					'left': chunk2['left'],
					'right': chunk1['right'],
					'combinator': '\\'
				}
			return(state)

def forward_crossed(chunk1, chunk2):
	if chunk1['combinator'] == '/' and chunk2['combinator'] == '\\':
		if chunk1['right'] == chunk2['left']:
			state = {
					'left': chunk1['left'],
					'right': chunk2['right'],
					'combinator': '\\'
				}
			return(state)

def backward_crossed(chunk1, chunk2):
	if chunk1['combinator'] == '/' and chunk2['combinator'] == '\\':
		if chunk1['left'] == chunk2['right']:
			state = {
					'left': chunk2['left'],
					'right': chunk1['right'],
					'combinator': '/'
				}
			return(state)

def apply_all(tag, chunk=None):
	if chunk == None: 
		combined = tag
	else:
		combined = forward_appl(chunk, tag)
	if combined == None: combined = backward_appl(chunk, tag)
	if combined == None: combined = forward_harmonic(chunk, tag)
	if combined == None: combined = backward_harmonic(chunk, tag)
	if combined == None: combined = forward_crossed(chunk, tag)
	if combined == None: combined = backward_crossed(chunk, tag)

	return(combined)

def nested_combine(tag, chunk, side, nested_el):
	if nested_el == 'chunk':
		el = deepcopy(chunk[side])
		el = el.replace(' ', '')
		nested = re.findall(r'(\(.*\))', el)
	else:
		el = deepcopy(tag[side])
		el = el.replace(' ', '')
		nested = re.findall(r'(\(.*\))', el)

	#print(len(nested))
	if len(nested) > 0 and len(nested)<5:  # < 5 is aribitrary

		regex = r'(\/[A-Z]+\)$)'  #matches /XP) (so rules nested on left)
		matched = re.findall(regex, nested[0])
		combinator = '/'

		if len(matched) == 0: 
			regex = r'(\\[A-Z]+\)$)' # matches \\XP)
			matched = re.findall(regex, nested[0])
			combinator = '\\'

		if len(matched) == 0:
			regex = r'(^\([A-Z]+\/)'#matches (XP/  (so rules nested on right)
			matched = re.findall(regex, nested[0])
			combinator = '/'

		if len(matched) == 0:
			regex = r'(^\([A-Z]+\\)' #matches (XP\\
			matched = re.findall(regex, nested[0])
			combinator = '\\'

		if len(matched) == 0:
			regex = r'(\)\/\()' #matches )/(  so nested on both
			matched = re.findall(regex, nested[0])
			combinator = '/'

		if len(matched) == 0:
			regex = r'(\)\\\()' #matches )\(  so nested on both
			matched = re.findall(regex, nested[0])
			combinator = '\\'

		if len(matched) == 0:
			regex = r'(\w+\/)'  #matches XP/ (so rules not nested)
			matched = re.findall(regex, nested[0])
			combinator = '/'


		if len(matched) == 0:
			regex = r'(\w+\\)'  #matches XP\ (so rules not nested)
			matched = re.findall(regex, nested[0])
			combinator = '\\'

		if len(matched) == 0:
			# print('no match for nested rule', nested[0])
			# can happen if nested rule is not the correct one (i.e. trying to find nested rule on the wrong side. 
			return(None)


		# combinator = '/'   # start with assuming '/'
		states = re.split(regex, nested[0])		

		#states = [s for s in states if s != combinator]
		# if len(states) == 1:
		# 	combinator = '\\'
		# 	states = str.split(nested[0], combinator)

		#states = [s[:-1] for s in states if s[-1] == ')']
		states = [x for x in states if x != '']

		# states = [s[:-1] if s[-1] == ')' and s[0] != '(' else s for s in states]

		states = [balance_parens(s) for s in states]

		states = [s for s in states if len(s) > 0]
		states = [s[1:] if s[0] in ['\\', '/'] else s for s in states]
		states = [s[:-1] if s[-1] in ['\\', '/'] else s for s in states]

		states = [x for x in states if x != '']
		states = [x for x in states if x != '(']
		states = [x for x in states if x != ')']

		# while states[0][0] == '(':
		# 	states[0] = states[0][1:]

		# while states[-1][-1] == ')':
		# 	states[-1] = states[-1][:-1]
		
		
		if len(states) > 3:   # I don't think I need this for the simple grammar??

			
			# TO DO: Deal with this corner case properly. Right now I am ignoring this because this only happens in cases where the model is going down a horribly wrong path and has many nested rules. 

			# But this can change depending on if I have more complex grammars later. 
			# print('REACHED GREATER THAN 3')
			return(None)


			# ind = 0
			# first_state = ''
			# for i, item in enumerate(states):
			# 	if item == combinator:
			# 		ind = i
			# 		break
			# 	else:
			# 		first_state += item

			# second_state = ''
			# for i in range(ind, len(states)):
			# 	second_state += states[i]

			# states = [first_state, second_state]
			# print('states greater than 3', states)




		if len(states) == 3:  # I don't think I need this??
			states = [states[0], states[2]]

			if len(re.findall(r'([\\/]+)', states[0]))>0:
				states[0] = '(' + states[0] + ')'
			if len(re.findall(r'([\\/]+)', states[1]))>0:
				states[1] = '(' + states[1] + ')'


		if len(states) == 2: 
			# print('HELLO')

			curr_state = {'left': states[0],
					 'right': states[1],
					 'combinator': combinator}

			# print(curr_state)

			
			if nested_el == 'chunk':
				combined = combine(tag, curr_state)
			else:
				combined = combine(curr_state, chunk)

			return(combined)
		else:
			print('MORE OR LESS THAN TWO STATES')
			# for item in states:
			# 	if str



def balance_parens(rule):
	num_open = 0
	num_closed = 0

	for char in rule:
		if char == '(': num_open +=1
		if char == ')': num_closed +=1

	# if num_open == num_closed:
	# 	return(rule)

	if num_open > num_closed:
		diff = num_open-num_closed
		rule = rule[diff:]  #this assumes that all extra parens are in the front.


	if num_open < num_closed:
		diff = num_closed-num_open
		rule = rule[:-diff] #this assumes that all extra parens are in the end.

	return(rule)	

def add_parens(rule):
	# print('In balance_parens')
	rule = balance_parens(rule)

	nested = re.findall(r'(\.*[\//])', rule)
	if len(nested)>0:   # we want to add parens only for nested combined rules
		if rule[-1] != ')' or rule[0]!= '(':
			rule = '(' + rule + ')'

	return(rule)





def combine(tag, chunk):
	# print(tag)
	# print(chunk)
	combined = apply_all(tag, chunk)

	if combined != None:
		return(combined)

	else:  #try nesting
		nested_comb = nested_combine(tag,  chunk, side='right', nested_el='chunk')


		if nested_comb != None:
			nested_comb_rule =  nested_comb['left'] + nested_comb['combinator'] + nested_comb['right'] 

			nested_comb_rule = add_parens(nested_comb_rule)


			combined = {
				'left': chunk['left'],
				'right': nested_comb_rule,
				'combinator': chunk['combinator']
			}

			return(combined)

		nested_comb = nested_combine(tag, chunk, side='left', nested_el='chunk')


		if nested_comb != None:
			nested_comb_rule = nested_comb['left'] + nested_comb['combinator'] + nested_comb['right']
			nested_comb_rule = add_parens(nested_comb_rule)

			combined = {
				'left': nested_comb_rule,
				'right': chunk['right'],
				'combinator': chunk['combinator']
			}

			return(combined)


		nested_comb = nested_combine(tag, chunk, side='right', nested_el='tag')



		if nested_comb != None:
			nested_comb_rule = nested_comb['left'] + nested_comb['combinator'] + nested_comb['right']
			nested_comb_rule = add_parens(nested_comb_rule)

			combined = {
				'left': tag['left'],
				'right': nested_comb_rule,
				'combinator': tag['combinator']
			}

			return(combined)

		nested_comb = nested_combine(tag, chunk, side='left', nested_el='tag')
		

		if nested_comb != None:
			# print('REACHED HERE')
			nested_comb_rule = nested_comb['left'] + nested_comb['combinator'] + nested_comb['right']
			nested_comb_rule = add_parens(nested_comb_rule)

			combined = {
				'left': nested_comb_rule,
				'right': tag['right'],
				'combinator': tag['combinator']
			}

			#combined = nested_comb


			return(combined)


def train(actr_model, model_type, sents):
	num_prev_tags = sum(actr_model.base_count.values())

	if model_type == 'ep':
		supertag_sentence = supertag_sentence_ep
	else:
		supertag_sentence = supertag_sentence_wd

	for i,sent in enumerate(sents):
		# curr_tag_list = tags[i].split()

		final_state, tags, words = supertag_sentence_wd(actr_model, sents)


		ccg_tag_list = [(words[n], tags[n]) for n in range(len(word_list))]

		for j, pair in enumerate(ccg_tag_list):
			num_prev_tags += 1
			word = pair[0]
			tag = pair[1]

			actr_model.lexical_count[word][tag] +=1

			actr_model.base_count[tag] += 1

			actr_model.base_instance[tag].append(num_prev_tags)


with open('./declmem/syntax_chunks_ep.pkl', 'rb') as f:
	syntax_chunks_ep = pickle.load(f)

with open('./declmem/syntax_chunks_wd.pkl', 'rb') as f:
	syntax_chunks_wd = pickle.load(f)


def print_states(tag_list, word_list):
	goal_buffer = None
	for i,tag_label in enumerate(tag_list):
		tag = syntax_chunks_wd[tag_label]
		# print()
		# print('goal_buffer befpre', goal_buffer)
		goal_buffer = combine(tag, goal_buffer)
		
		# print('word,tag', word_list[i], tag)
		# print('goal_buffer after', goal_buffer)
		curr_rule = tag['left'] + tag['combinator'] + tag['right']
		curr_state = goal_buffer['left'] + goal_buffer['combinator'] + goal_buffer['right']
		print(word_list[i], curr_rule, curr_state)



# tag = {'left': '(TP\\DP)', 'right': 'DP', 'combinator': '/'}

# chunk = {'left': ' (TP/(TP\\DP))', 'right': '', 'combinator': ''}



# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP', '.']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat', 'eos']


# chunk = {'left': 'TP', 'right': '(TP\\DP)', 'combinator': '/'}

# print(combine(tag, chunk))

# x/y x 

# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP', '.']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat', 'eos']

# print_states(tag_list, word_list)

# print('-----------------------------')


# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_loc', 'Prep', 'Det', 'Adj', 'NP', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'arrived', 'at', 'the', 'unreliable', 'palace', '.']

# print_states(tag_list, word_list)

# print('-----------------------------')

# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'aux', 'Adj', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'was', 'unreliable', '.']

# print_states(tag_list, word_list)

# print('-----------------------------')



# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'V_intrans', 'Adv', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'sang', 'beautifully', '.']

# print_states(tag_list, word_list)

# print('-----------------------------')


# tag_list = ['Det', 'NP_VoiceP', 'Vt_pass', 'Prep', 'Det', 'NP', 'aux', 'Adj', 'eos']

# word_list = ['the', 'defendant', 'examined', 'by', 'the', 'lawyer', 'was', 'unreliable', '.']

# print_states(tag_list, word_list)

# print('-----------------------------')


# tag_list = ['Det', 'NP_CP_null', 'compsubj_null', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# print_states(tag_list, word_list)

# print('-----------------------------')

# tag_list = ['Det', 'NP_VoiceP', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# word_list = ['the', 'defendant', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# print_states(tag_list, word_list)

# print('-----------------------------')

# word_list = ['the', 'defendant', 'who', 'was', 'being','examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_prog', 'ProgP', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# print_states(tag_list, word_list)

# print('-----------------------------')


# word_list = ['the', 'defendant', 'being','examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_ProgP', 'ProgP', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP']

# print_states(tag_list, word_list)

# print('-----------------------------')


# word_list = ['the', 'defendant', 'who', 'examined','the', 'lawyer', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP', 'compsubj', 'Vt_act',  'Det', 'NP', 'Vt_act', 'Det', 'NP']

# print_states(tag_list, word_list)

# print('-----------------------------')

# word_list = ['the', 'defendant', 'who','the', 'lawyer', 'examined', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP', 'compobj', 'Det', 'NP', 'Vt_act', 'Vt_act', 'Det', 'NP']

# print_states(tag_list, word_list)



# word_list = ['the', 'defendant', 'who_del', 'the', 'lawyer', 'examined', 'liked', 'the', 'cat']

# tag_list = ['Det', 'NP_CP_null', 'compobj_null', 'Det', 'NP', 'Vt_act', 'Vt_act', 'Det', 'NP']


# tag_list = ['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP', 'Vt_act', 'Det', 'NP', 'conj', 'V_intrans', 'Adv', 'eos']

# word_list = ['the', 'defendant', 'who', 'was', 'examined', 'by', 'the', 'lawyer', 'liked', 'the', 'cat', 'and',  'sang', 'unreliably', '.']

# print_states(tag_list, word_list)

# print('-----------------------------')

## This is wrong parse


# tag_list =['Det', 'NP_CP', 'compsubj', 'aux_pass', 'Vt_pass', 'Prep', 'Det', 'NP']


# word_list = ['the', 'defendant', 'who', 'was', 'loved', 'by', 'his', 'tenants']



# print_states(tag_list, word_list)


# print(add_parens('((TP\\DP)/DP)/NP'))

# print(add_parens('((TP\\DP)/DP)/NP)))'))

# print(add_parens('(((TP\\DP)/DP)/NP)'))
