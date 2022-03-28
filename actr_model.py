import numpy as np
from copy import deepcopy

class actr_model:
	def __init__(self,decay, max_activation, prior_weight, time_factor, syntax_chunks, lexical_chunks, supertag_function):

		self.base_count = {key:0 for key in syntax_chunks}

		self.base_instance = {key:[] for key in syntax_chunks} 

		self.base_act = {key:0 for key in syntax_chunks}

		self.lexical_count = {key:{key2: 0 for key2 in syntax_chunks} for key in lexical_chunks}

		self.lexical_act = {key:{key2: 0 for key2 in syntax_chunks} for key in lexical_chunks}

		self.decay = decay

		self.max_activation = max_activation

		self.prior_weight = prior_weight

		self.time_factor = time_factor

		self.syntax_chunks = syntax_chunks

		self.lexical_chunks = lexical_chunks

		self.supertag_sentence = supertag_function


	def compute_baseact(self, tag):
		curr_tag_instance = np.array(deepcopy(self.base_instance[tag]))

		if len(curr_tag_instance) == 0:
			activation = 0
		else:
			num_total_tags = sum(self.base_count.values())

			time_seq = (num_total_tags+1)-curr_tag_instance
			time_seq = time_seq*self.time_factor   
			activation = np.log(sum(np.power(time_seq, -self.decay)))

		return(max(0,activation))

	def update_base_activation(self):
		for tag in self.base_act:
			self.base_act[tag] = self.prior_weight*self.compute_baseact(tag)

	def update_lexical_activation(self):
		for word, word_dict in self.lexical_count.items():
			word_sum = sum(word_dict.values())
			for tag in word_dict:
				if word_sum == 0: 
					prob = 0
				else:
					prob = word_dict[tag]/word_sum
				self.lexical_act[word][tag] += prob*self.max_activation

	def update_counts(self, sents):
		num_prev_tags = sum(self.base_count.values())

		for sent in sents:
			# print(sent)
			final_state, tags, words = self.supertag_sentence(self, sent)

			ccg_tag_list = [(words[n], tags[n]) for n in range(len(words))]

			# print(sent)
			# print(tags)
			# print()

			for j, pair in enumerate(ccg_tag_list):
				num_prev_tags += 1
				word = pair[0]
				tag = pair[1]

				self.lexical_count[word][tag] +=1
				self.base_count[tag] += 1
				self.base_instance[tag].append(num_prev_tags)



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




