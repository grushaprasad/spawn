import re
import math
import random
from copy import deepcopy
import numpy as np
import pickle
import ccg


## Currently show that this can work with just base-level. I want to eventually add inhibition and lexical activation. 
def predict_null(word, curr_tag, curr_parse_state, null_base, syntax_chunks, tr_rules, noise_sd):
    """
    Do I need inhibition ?? Do I say when I reanalyze, I inhibit reanalyze the decision for null as well? Do I really want to split the is_null and get_null?  
    """
    found_valid = False
    poss_tags = list(null_base.keys())

    while not found_valid: 
        ## initialize with base-level activation
        act_dict = {tag: null_base[curr_tag][tag] for tag in poss_tags}

        choice = max(curr_act_dict, key=lambda key:curr_act_dict[key])

        ## add in noise 
        for tag in curr_act_dict:  
            curr_act_dict[tag] += np.random.normal(0, noise_sd)

        null_pred = max(curr_act_dict, key=lambda key:curr_act_dict[key])

        if null_pred == 'not-null':
            found_valid = True
        else:
            null_chunk = syntax_chunks[choice]

            combined = ccg.combine(
                tag = null_chunk,
                parse_state = parse_state,
                tr_rules = tr_rules)

            if combined:
                found_valid = True
            else: ## invalid tag with parse state
                poss_tags.remove(null_pred) 
    
    return null_pred







        




    


def predict_null2(word, curr_tag, null_base, null_lexical, curr_inhibition, curr_time):
    """
    Returns just decision of null or not-null. Additional step of figuring out which specific null has to exist. 

    For inhibition, we need to figure out two things: 

    """
    pass


# def supertag_sentence(sentence):
#     words = sentence.split()
#     num_iters = 0
#     i = 0

#     parse_states = [None]

#     while(i < len(words) and num_iters < max_iters):
#         num_iters +=1
#         curr_word = words[i]
#         poss_tags = deepcopy(lexical_chunks[curr_word]['syntax'])


#         found_valid_tag = False

#         curr_tag = generate_valid_supertag(word. inhibition) # returns None if no valid tag

#         if curr_tag:
#             curr_tag_chunk = syntax_chunks[curr_tag]

#             curr_parse_state = ccg.combine(tag = curr_tag_chunk,
#                 parse_state = parse_states[-1],
#                 tr_rules = tr_rules)

#             null_pred = predict_null(tag, word) #retu
#             if null_pred != 'not-null': #there is null
#                 words.insert(i+1, null_pred)




#         else:
#             reanalyze()










        





lexical_chunks = {
    'the': {'syntax': ['Det']}

}

syntax_chunks = {
    'Det': {
        'left': 'NP',
        'right': 'N',
        'combinator': '/'
    },
    'Noun': {
        'left': 'N',
        'right': '',
        'combinator': ''
    },
    'SC_verb':{
        'left': '(S\\NP)',
        'right': 'S',
        'combinator': ''
    },
    'Tr_verb':{
        'left': '(S\\NP)',
        'right': 'NP',
        'combinator': '/'
    },
    'that1':{
        'left': 'S',
        'right': 'S',
        'combinator': '/'
    },
    'that2':{
        'left': '(N/N)',
        'right': '(S/NP)',
        'combinator': '/'
    }

}

target = 'the director announced'


vpc_primes = {
    'that': 'the professor thought that the students appreciated the idea',
    'no-that': 'the professor thought the students appreciated the idea'
}

rc_primes = {
    'that': 'the professor appreciated the ideas that the students expressed',
    'no-that': 'the professor appreciated the ideas the students expressed'
}



null_categories = {
    'NULL-THAT1': syntax_chunks['that1'],
    'NULL-THAT2': syntax_chunks['that2']
}

null_base = {
    'Det':{
        'NULL-THAT1': 0,
        'NULL-THAT2': 0,
        'not-null': 3
    },


}

null_act1 = {
    'NULL-THAT1': 1,
    'NULL-THAT2': 0.4,
    'not-null': 3
}

null_base1 = {}

for key in syntax_chunks: 
    null_base1[key] = {k:0 for k in null_categories}
    null_base1[key]['not-null'] = 0

null_base2 = {key: {'null':0, 'not-null':0} for key in syntax_chunks}



# print(null_base)

## Question: when given the target, is there a null predicted after the target? 


## Question: if I adopt the notion that you predict a specific null element, and then allow for reanalysis, then will this allow for cross-structural priming? 


def convert_to_prob(d):
    total = sum(d.values())
    return {key: val/total for key,val in d.items()}


d1 = {
    'NULL-THAT1': 1,
    'not-null': 3
}

d2 = {
    'NULL-THAT2': 0.4,
    'not-null': 3
}
print(convert_to_prob(null_act1))
print(convert_to_prob(d1))
print(convert_to_prob(d2))

## So yes, when we remove the impossible one, there is more chance for the other one to be picked. Not even sure if I need reanalysis here for incorrect null pred? I think the valid_supertag can handle this? 


## What I want to do: 
## Show that in principle the null mechanism can result in more prob of null being generated even when different null is primed. Maybe I sample and show what happens when I sample?  

## Show what happens when we prime --> basically increase base-level for one of the things. 

## Also think through: 
## How to translate the comprehension to production ? 










