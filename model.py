import numpy as np
from copy import deepcopy
import math
import supertagger

class actr_model:
	# def __init__(self,decay, max_activation, noise_sd, latency_factor, latency_exponent, syntax_chunks, lexical_chunks, type_raising_rules, null_mapping supertag_function):
	def __init__(self,decay, max_activation, noise_sd, latency_factor, latency_exponent, syntax_chunks, lexical_chunks, type_raising_rules, null_mapping, max_iters, reanalysis_type, temperature):

		eps = 0.000001

		self.base_count = {key:0 for key in syntax_chunks}

		self.base_instance = {key:[] for key in syntax_chunks} 

		self.base_act = {key:0 for key in syntax_chunks}

		self.lexical_count = {key:{key2: 0 for key2 in syntax_chunks} for key in lexical_chunks}

		self.lexical_act = {key:{key2:0 for key2 in syntax_chunks} for key in lexical_chunks}


		self.time = 0

		self.decay = decay

		self.max_activation = max_activation

		self.noise_sd = noise_sd

		self.lf = latency_factor

		self.le = latency_exponent

		self.syntax_chunks = syntax_chunks

		self.lexical_chunks = lexical_chunks

		self.type_raising_rules = type_raising_rules

		self.null_mapping = null_mapping

		self.lexical_null_count = {key:{key2: 0 for key2 in list(null_mapping.values())+['not-null']} for key in lexical_chunks}

		self.lexical_null_act = {key:{key2: 0 for key2 in list(null_mapping.values())+['not-null']} for key in lexical_chunks}

		self.num_failed_sents = 0

		self.num_retries = 0

		self.failed_sents = []

		self.max_iters = max_iters

		self.eps = eps

		self.reanalysis_type = reanalysis_type

		self.temperature = temperature

		#self.supertag_sentence = supertag_function


	# def compute_inhibition(self, )


	def compute_baseact(self, tag):
		curr_tag_instance = np.array(deepcopy(self.base_instance[tag]))

		if len(curr_tag_instance) == 0:
			activation = 0
		else:
			num_total_tags = sum(self.base_count.values())

			#time_seq = (num_total_tags+1)-curr_tag_instance
			#time_seq = time_seq*self.time_factor   
			time_seq = self.time + self.eps - curr_tag_instance  #if we don't add eps the last tag will be 0
			# print('time_seq',time_seq)
			activation = sum(np.power(time_seq, -self.decay))

		if activation <= 0: # cannot log zero or negative 
			return 0
		else:
			return max(0, np.log(activation)) # do not want negative activation

	def compute_inhibition(self, tag, inhibition_list, curr_time):
		inhibition_val = 0
		for item in inhibition_list:
			if item['tag'] == tag:
				time_since_item = curr_time + self.eps - item['time']
				inhibition_val += np.power(time_since_item, -self.decay)

		if inhibition_val <= 0:  # cannot log zero or negative 
			return 0
		else:
			return max(0,np.log(inhibition_val)) # do not want negative inhibition

	def update_base_activation(self):
		# print(self.base_act)
		for tag in self.base_act:
			self.base_act[tag] = self.compute_baseact(tag)

	def update_lexical_activation(self):
		# update activation from words to tags
		for word, word_dict in self.lexical_count.items():
			word_sum = sum(word_dict.values())
			for tag in word_dict:
				if word_sum == 0: 
					prob = 0
				else:
					prob = word_dict[tag]/word_sum

				self.lexical_act[word][tag] = prob*self.max_activation

		## update activation from words to null
		for word, word_dict in self.lexical_null_count.items():
			word_sum = sum(word_dict.values())
			for null_el in word_dict:
				if word_sum == 0:
					prob = 0
				else:
					prob = word_dict[null_el]/word_sum
				self.lexical_null_act[word][null_el] = prob*self.max_activation

	def convert_to_rt(self,act_list):
		rt = 0
		for val in act_list:
			rt += self.lf*math.exp(-(self.le*val))
		return(rt)

	def update_counts(self, sents):
		num_prev_tags = sum(self.base_count.values())

		for sent in sents:
			# print(sent)
			self.num_retries = 0 #set the number of retries for any sentence to be zero. 
			final_state, tags, words, act_vals = supertagger.supertag_sentence(self, sent, use_priors=True)
			if final_state == None: #if model "gave up" after max tries
				self.num_failed_sents +=1 #keep track of how many sentences its retried. 
				self.failed_sents.append(sent)
				# print(f'{sent} not parsed')

			while final_state == None: # keep trying till a parse is found
				self.num_retries +=1
				final_state, tags, words, act_vals = supertagger.supertag_sentence(self, sent, use_priors=False)

			if self.num_retries!=0:
				print('num retries', self.num_retries, sent)
			# print(tags)
			# print(len(act_vals[0]))
			# print(words[1], act_vals[1])

			
			# print(sent)
			# print(tags)
			# print()
			if final_state != None:
				ccg_tag_list = [(words[n], tags[n], act_vals[n]) for n in range(len(words))]

				for j, pair in enumerate(ccg_tag_list):
					num_prev_tags += 1
					word = pair[0]
					tag = pair[1]
					act = pair[2]
					if j < len(ccg_tag_list)-1: #not last word
						next_word = ccg_tag_list[j+1][0]
					else:
						next_word = None
					
					time_for_tag = self.convert_to_rt(act)

					self.time += time_for_tag + 0.05  #50 ms for production rule firing 
					
					self.lexical_count[word][tag] +=1
					if next_word:  #i.e. if there is a next word
						if next_word in self.null_mapping.values():
							self.lexical_null_count[word][next_word] +=1
						else:
							self.lexical_null_count[word]['not-null'] +=1
					self.base_count[tag] += 1
					#self.base_instance[tag].append(num_prev_tags)
					self.base_instance[tag].append(self.time)







