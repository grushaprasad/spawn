import numpy as np
from copy import deepcopy
import math
import supertagger

class actr_model:
	# def __init__(self,decay, max_activation, noise_sd, latency_factor, latency_exponent, syntax_chunks, lexical_chunks, type_raising_rules, null_mapping supertag_function):
	def __init__(self,decay, max_activation, noise_sd, latency_factor, latency_exponent, syntax_chunks, lexical_chunks, type_raising_rules, null_mapping, max_iters):

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

		self.num_retried_sents = 0

		self.num_retries = 0

		self.max_iters = max_iters

		self.eps = eps

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
			activation = np.log(sum(np.power(time_seq, -self.decay)))
			# activation = sum(np.power(time_seq, -self.decay))

		# if activation == 0:
		# 	return 0
		# else:
		# 	return np.log(activation)

		return(max(0,activation))  # should this be 0 ir should this be eps?

	def compute_inhibition(self, tag, inhibition_list, curr_time):
		inhibition_val = 0
		for item in inhibition_list:
			if item['tag'] == tag:
				time_since_item = curr_time + self.eps - item['time']
				inhibition_val += np.power(time_since_item, -self.decay)

		# if inhibition_val == 0:
		# 	return 0
		# else:
		# 	return np.log(inhibition_val)
		return(max(0,np.log(inhibition_val)))  # should this be 0 ir should this be eps?

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
				#self.lexical_act[word][tag] += prob*self.max_activation  #Why was I adding before??? 
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
			final_state, tags, words, act_vals = supertagger.supertag_sentence(self, sent)
			if final_state == None:
				self.num_retried_sents +=1 #keep track of how many sentences its retried. 

			while final_state == None: #if supertag fails because of hitting max tries
				self.num_retries +=1
				final_state, tags, words, act_vals = supertagger.supertag_sentence(self, sent)
				# if self.num_retries > 20:
				# 	final_state, tags, words, act_vals = supertagger.supertag_sentence(self, sent, print_stages=True)
				# else:
				# 	final_state, tags, words, act_vals = supertagger.supertag_sentence(self, sent)
			# print(tags)
			# print(len(act_vals[0]))
			# print(words[1], act_vals[1])

			ccg_tag_list = [(words[n], tags[n], act_vals[n]) for n in range(len(words))]

			# print(sent)
			# print(tags)
			# print()

			for j, pair in enumerate(ccg_tag_list):
				num_prev_tags += 1
				word = pair[0]
				tag = pair[1]
				act = pair[2]
				if j < len(ccg_tag_list)-1: #not last word
					next_word = ccg_tag_list[j+1][0]
				else:
					next_word = None
				#print(len(act))
				#print(word, tag, act)

				# time_for_tag = 0
				# for x in act:
				# 	time_for_tag += self.convert_to_rt(x)

				#times = [self.convert_to_rt(x) for x in act]
				
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




	# def update_counts(self, sents, tags):
	# 	num_prev_tags = sum(self.base_count.values())

	# 	for i,sent in enumerate(sents):
	# 		curr_tag_list = tags[i].split()
	# 		word_list = sent.split()
	# 		ccg_tag_list = [(word_list[n], curr_tag_list[n]) for n in range(len(word_list))]

	# 		for j, pair in enumerate(ccg_tag_list):
	# 			num_prev_tags += 1
	# 			word = pair[0]
	# 			tag = pair[1]

	# 			self.lexical_count[word][tag] +=1

	# 			self.base_count[tag] += 1

	# 			self.base_instance[tag].append(num_prev_tags)



# def create_lexical_chunks(stimuli, model_type):
# 	pos_dict = {
# 		'rrc': ['det', 'noun', 'rcv', 'prep', 'det', 'noun', ]
# 	}




