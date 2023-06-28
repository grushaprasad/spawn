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
        # chunk2_rule = '(' + chunk2['left'] + chunk2['combinator'] + chunk2['right'] + ')'
        chunk2_rule = make_rule(chunk2)
        # print('reached here forward_appl')
        # print(chunk2_rule)

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
        # chunk1_rule = '(' + chunk1['left'] + chunk1['combinator'] + chunk1['right'] + ')'
        # print(chunk1_rule)
        chunk1_rule = make_rule(chunk1)

        if chunk2['combinator'] == '\\' and chunk1_rule == chunk2['right']:
            state = {
                'left': chunk2['left'],
                'right': '',
                'combinator': ''
            }
            print('reached here')

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

def apply_all(tr_rules, tag, parse_state=None, print_rule=False):
    if parse_state == None: 
        combined = tag
        return tag
    else:
        combined = forward_appl(parse_state, tag)
        applied_rule = 'forward'
        if combined == None:
            combined = backward_appl(parse_state, tag)
            applied_rule = 'backward'
        if combined == None:
            combined = forward_harmonic(parse_state, tag)
            applied_rule = 'forward_harmonic'
        if combined == None:
            combined = backward_harmonic(parse_state, tag)
            applied_rule = 'backward_harmonic'
        if combined == None: 
            combined = forward_crossed(parse_state, tag)
            applied_rule = 'forward_crossed'
        if combined == None: 
            combined = backward_crossed(parse_state, tag)
            applied_rule = 'backward_crossed'
    
        if combined != None:
            if print_rule:
                print('applied', applied_rule)
            return combined
        else:
            tr_tags = type_raise(tag, tr_rules)
            for tr_tag in tr_tags:
                combined = apply_all(tr_rules,tr_tag, parse_state)
                if combined != None:
                    return combined

            # Next try to type raise the chunk (not sure if this is required)
            tr_tags = type_raise(parse_state, tr_rules)
            for tr_tag in tr_tags:
                combined = apply_all(tr_rules, tag, tr_tag)
                if combined != None:
                    return combined


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
        #   combinator = '\\'
        #   states = str.split(nested[0], combinator)

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
        #   states[0] = states[0][1:]

        # while states[-1][-1] == ')':
        #   states[-1] = states[-1][:-1]
        
        
        if len(states) > 3:   # I don't think I need this for the simple grammar??

            
            # TO DO: Deal with this corner case properly. Right now I am ignoring this because this only happens in cases where the model is going down a horribly wrong path and has many nested rules. 

            # But this can change depending on if I have more complex grammars later. 
            # print('REACHED GREATER THAN 3')
            return(None)


            # ind = 0
            # first_state = ''
            # for i, item in enumerate(states):
            #   if item == combinator:
            #       ind = i
            #       break
            #   else:
            #       first_state += item

            # second_state = ''
            # for i in range(ind, len(states)):
            #   second_state += states[i]

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
            #   if str



def balance_parens(rule):
    num_open = 0
    num_closed = 0

    for char in rule:
        if char == '(': num_open +=1
        if char == ')': num_closed +=1

    # if num_open == num_closed:
    #   return(rule)

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

    # nested = re.findall(r'(\.*[\//])', rule)
    if len(rule.split('\\')) > 1 or len(rule.split('/')) > 1:
    # if len(nested)>0:   # we want to add parens only for nested combined rules
        if rule[-1] != ')' or rule[0]!= '(':
            rule = '(' + rule + ')'

    return(rule)

# def to_recurse(chunk, type):

def get_nested(chunk, chunk_type):
    """
    DP/(TP/DP)) + DP/NP -> DP/(TP/NP)
    DP/(TP\\DP)/DP) + DP/NP -> DP/(TP\\DP)/NP)

    For goal buffers:
    - DP/(TP/DP) : {'left': TP, 'right': DP, 'combinator': '/'} 
    - DP/((TP\\DP)/DP) : True
    - DP\\(TP/DP) : False
    - DP/TP : False
    
    To do: think about what needs to happen if tag is nested. 

    For now, make this work for easy cases where I would need it. 
    Worry about the general cases later. 
    """

    if chunk_type == 'parse_state':
        if chunk['combinator'] == '/' and chunk['right'][-1] == ')':
            split = chunk['right'].split('/')
            left = balance_parens('/'.join(split[:-1])).strip()
            right = balance_parens(split[-1]).strip()
            return {'left': left, 'right': right, 'combinator':'/'}
        # elif chunk['combinator'] == '' and chunk['right'] == '' and chunk['left'][-1]== ')': #a nested rule in the left but leaving the right and combinator empty
        #   split = chunk['left'].split('/')
        #   left = balance_parens('/'.join(split[:-1])).strip()
        #   right = balance_parens(split[-1]).strip()
        #   return {'left': left, 'right': right, 'combinator':'/'}

            
def fix_chunk(chunk):
    """
    Sometimes we end up with a chunk like this:
    {'left': '(TP/(TP\\DP))', 'right': '', 'combinator': ''}

    We want that to instead be:
    {'left': 'TP', 'right': '(TP\\DP)', 'combinator: '/'}

    TO DO: 
    What if the combinator was \\ ? 
    (TP\\(TP\\DP)) -> {'left': 'TP', 'right': '(TP\\DP)', 'combinator: '\\'}

    ((TP\\DP)/TP) -> {'left': '(TP\\DP)', 'right': 'TP', 'combinator: '/'}

    ((TP/DP)\\TP) -> {'left': '(TP/DP)', 'right': 'TP', 'combinator: '\\'}

    I need to find the primary combinator and split by that. 

    """
    if chunk != None and chunk['combinator'] == '' and chunk['right'] == '' and chunk['left']!= '' and chunk['left'][-1]== ')':
        split = chunk['left'].split('/')
        left = balance_parens('/'.join(split[:-1])).strip()
        right = balance_parens(split[-1]).strip()

        return {'left': left, 'right': right, 'combinator':'/'}
    else:
        return chunk


def combine(parse_state, tag, tr_rules, print_rule = False):
    ## Make sure goal buffer is in the right format
    parse_state = fix_chunk(parse_state)

    combined = apply_all(tr_rules, tag, parse_state, print_rule)
    if combined != None:
        return(combined)
    else:
        ## Try to see if the goal buffer is nested
        nested = get_nested(parse_state, 'parse_state')
        if nested != None:
            nested_combined = apply_all(tr_rules, tag, nested, print_rule)
            if nested_combined != None:

                combined = {
                    'right': make_rule(nested_combined),
                    'left': parse_state['left'],
                    'combinator': '/'
                }
                return combined

        ## To do: think about cases where 


def type_raise(chunk, type_raising_rules):
    rule = make_rule(chunk)
    #return the chunks associated with rule or empty list
    return type_raising_rules.get(rule, []) 


def make_rule(chunk):
    if chunk != None:
        return  add_parens(chunk['left'] + chunk['combinator'] + chunk['right']) 
    else:
        return ''

def test_combine(tr_rules):
    #Forward composition
    combined = combine({'left': 'DP',
                        'right': 'NP',
                        'combinator': '/'},
                       {'left': 'NP',
                        'right': '',
                        'combinator': ''},
                        tr_rules)
    assert make_rule(combined) == 'DP'

    # Backward composition
    combined = combine({'left': 'DP',
                        'right': '',
                        'combinator': ''},
                       {'left': 'TP',
                        'right': 'DP',
                        'combinator': '\\'},
                        tr_rules)
    assert make_rule(combined) == 'TP'

    # Forward harmonic composition
    combined = combine({'left': 'DP',
                        'right': 'VoiceP',
                        'combinator': '/'},
                       {'left': 'VoiceP',
                        'right': 'PP',
                        'combinator': '/'},
                        tr_rules)
    assert make_rule(combined) == '(DP/PP)'

    # Backward harmonic compoisition
    combined = combine({'left': 'TP',
                        'right': 'DP',
                        'combinator': '\\'},
                       {'left': 'eos',
                        'right': 'TP',
                        'combinator': '\\'},
                        tr_rules)
    assert make_rule(combined) == '(eos\\DP)'

    # Forward crossed composition
    combined = combine({'left': 'CP',
                        'right': 'TP',
                        'combinator': '/'},
                       {'left': 'TP',
                        'right': 'DP',
                        'combinator': '\\'},
                        tr_rules)
    assert make_rule(combined) == '(CP\\DP)'

    # Backward corssed composition
    combined = combine({'left': 'TP',
                        'right': 'VoiceP',
                        'combinator': '/'},
                       {'left': 'eos',
                        'right': 'TP',
                        'combinator': '\\'},
                        tr_rules)
    assert make_rule(combined) == '(eos/VoiceP)'

    # Type raising of DP
    combined = combine({'left': 'DP',
                        'right': '',
                        'combinator': ''},
                       {'left': '(TP\\DP)',
                        'right': 'DP',
                        'combinator': '/'},
                        tr_rules)
    assert make_rule(combined) == '(TP/DP)'

    # Nested rules
    combined = combine({'left': 'DP',
                        'right': '(VoiceP/ProgP)',
                        'combinator': '/'},
                       {'left': 'ProgP',
                        'right': '',
                        'combinator': ''},
                        tr_rules)
    assert make_rule(combined) == '(DP/VoiceP)'

    combined = combine({'left': 'DP',
                        'right': '(TP\\DP)',
                        'combinator': '/'},
                       {'left': '(TP\\DP)',
                        'right': 'DP',
                        'combinator': '/'},
                        tr_rules)
    assert make_rule(combined) == '(DP/DP)'

    combined = combine({'left': '(TP\\DP)',
                        'right': '(VoiceP/ProgP)',
                        'combinator': '/'},
                       {'left': 'ProgP',
                        'right': '',
                        'combinator': ''},
                        tr_rules)
    assert make_rule(combined) == '((TP\\DP)/VoiceP)'

    combined = combine({'left': 'DP',
                        'right': '(((TP\\DP)/DP)/DP)',
                        'combinator': '/'},
                       {'left': 'DP',
                        'right': 'NP',
                        'combinator': '/'},
                        tr_rules)
    assert make_rule(combined) == '(DP/(((TP\\DP)/DP)/NP))'

    combined = combine({'left': 'DP',
                        'right': '((TP\\DP)/DP)',
                        'combinator': '/'},
                       {'left': '(TP\\DP)',
                        'right': 'DP',
                        'combinator': '/'},
                        tr_rules)
    assert make_rule(combined) == 'DP'

    combined = combine({'left': '(TP/(TP\\DP))',
                        'right': '',
                        'combinator': ''},
                       {'left': '(TP\\DP)',
                        'right': 'DP',
                        'combinator': '/'},
                        tr_rules)
    assert make_rule(combined) == '(TP/DP)'

        ## TO DO: write tests for when it should fail

if __name__ == '__main__':
    import pickle
    with open('./declmem/type_raising_rules.pkl', 'rb') as f:
        type_raising_rules = pickle.load(f)
        print('Running tests')
    test_combine(type_raising_rules)