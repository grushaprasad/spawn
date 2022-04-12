import pickle

animate_nouns = list(set(['prince', 'duke', 'children', 'guardian',  'toddler', 'parents',  'singer', 'fans', 'princess', 'magician', 'employee', 'manager', 'doctor', 'nurse','student', 'teacher', 'clerk', 'spy', 'agent', 'pharmacist', 'patient', 'troops', 'terrorists', 'lion', 'hunter', 'senator', 'assassin', 'analyst', 'client', 'politician', 'party', 'king', 'genie', 'businessman', 'partner', 'executive', 'accountant', 'policeman', 'kidnapper', 'witch', 'peasant', 'suspect', 'investigator', 'prisoners', 'colonel', 'dog', 'boy', 'monkey', 'hatter', 'thief', 'officer', 'dentist', 'son', 'captain', 'team', 'actress', 'director', 'media', 'governor', 'president', 'valedictorian', 'family', 'widow', 'teenager', 'friend', 'woman', 'daughter', 'athlete', 'coach', 'boy', 'lady', 'secretaty', 'visitor', 'chemist', 'writer', 'physicist', 'journalist', 'gymnast', 'coach', 'musician', 'crowd', 'teacher', 'students', 'hairdresser', 'customer', 'defendant', 'lawyer', 'engineer', 'licensor', 'industrialist', 'auditor', 'patient', 'hygenist', 'thief', 'victim', 'warden', 'prisoner', 'student', 'researcher', 'electrician', 'company', 'landlord', 'tenants', 'comedian', 'audience', 'kid', 'janitor', 'employees', 'cashier' ,'customer','organization', 'government', 'broker', 'company', 'entrepreneur', 'industrialist', 'man', 'spy', 'actor', 'reporter', 'poet', 'writer', 'driver', 'passenger', 'priest', 'mob', 'management', 'employee', 'dancer', 'trainer', 'influencer', 'fans', 'girl', 'cat', 'toddler', 'dog', 'stranger', 'applicant', 'company', 'professor', 'university', 'chef', 'restaurant', 'developer', 'team', 'culprit', 'investigator', 'apprentice', 'artist', 'painter', 'student', 'robot', 'illustrator', 'soldier', 'general', 'writer', 'scholar', 'blacksmith', 'scholars', 'psychologist', 'officer', 'assistant', 'principal', 'administration', 'man', 'sister', 'plumber', 'carpenter', 'speaker', 'chairman', 'producer', 'association', 'elf', 'magician', 'therapist', 'doctor', 'protestors', 'authority', 'scientist', 'journalist', 'dragon', 'wizard', 'programmer', 'supervisor', 'woman', 'media', 'celebrity', 'builder', 'architect', 'secretary']))


location_nouns = list(set(['palace', 'park', 'store', 'restaurant', 'univeristy', 'construction-site', 'forest', 'hospital', 'home', 'street', 'streets']))

inanimate_nouns = list(set(['diamond', 'machine', 'evaluation', 'exam', 'report', 'pills', 'answers', 'allegations', 'information', 'ball', 'hats', 'public-statement', 'articles', 'contest', 'piece', 'donuts', 'test', 'money', 'walnuts', 'project', 'mission', 'contract', 'company', 'information', 'dish', 'problem', 'book', 'shield', 'job', 'car', 'awards', 'invention', 'mistake', 'table', 'canvas']))

other_nouns = list(set(['trance', 'losses', 'tax-fraud', 'teeth-extraction', 'effort', 'afternoon', 'hairstyle']))

determiners = list(set(['the', 'a', 'some', 'his', 'her', 'its', 'a-lot-of', 'more', 'many', 'their', 'an']))


adjectives = list(set(['happy', 'beautiful', 'unreliable', 'terrified', 'interested', 'seminal', 'uncomfortable', 'competent', 'considerate', 'cheerful', 'sunny', 'attentive', 'well-known', 'good', 'blue', 'heavy', 'innocent', 'famous', 'better', 'breathtaking', 'new', 'interesting', 'famous', 'successful', 'signature', 'radical', 'flawless']))

adverbs = list(set(['happily', 'beautifully', 'unreliably', 'joyfully', 'patiently', 'fearfully', 'enthusiastically', 'cheerfully', 'energetically', 'in-surprise', 'loudly', 'carefully', 'diligently', 'at-night', 'suddenly', 'playfully', 'excitedly', 'eventually', 'sullenly', 'wistfully', 'gleefully','peacefully', 'rapidly', 'in-surpise', 'efficiently', 'incessantly', 'in-fear', 'inaudibly', 'valiantly']))

aux = list(set(['was', 'looked', 'remained', 'were', 'felt']))

prepositions = list(set(['on', 'to', 'into', 'by', 'at', 'in', 'down']))

DP = list(set(['something', 'non-violence', 'privacy', 'plans-to-move', 'popularity', 'everyone']))


ambig_verbs = list(set(['accompanied', 'admired', 'approached', 'attacked', 'betrayed',
'captured', 'chased', 'congratulated', 'consoled', 'described', 'encouraged', 'examined', 'identified', 'loved', 'paid', 'recognized', 'recorded', 'scratched', 'selected', 'sketched', 'studied', 'envied', 'introduced', 'challenged', 'visited', 'recommended', 'arrested',  'threatened', 'made', 'promised', 'published', 'performed', 'proposed', 'signed', 'founded', 'advertised', 'solved', 'forged', 'bought']))


other_verbs = list(set(['received', 'passed', 'submitted', 'searched', 'suffered', 'refused', 'denied', 'lost', 'committed', 'stole', 'brought', 'tried', 'hid', 'cracked', 'impersonated', 'muttered', 'expanded', 'preached', 'talked-about', 'enjoyed', 'gained', 'managed', 'rolled-up', 'gained', 'wrote', 'quit', 'charmed', 'won', 'demonstrated', 'accepted', 'searched-for', 'divulged','played-with',]))

trans_verbs = ambig_verbs + other_verbs

intrans_verbs = list(set(['danced', 'sang', 'cackled', 'stopped', 'jumped', 'beamed', 'glowed', 'waited', 'ran-away', 'waved', 'smiled', 'grinned', 'practiced', 'exclaimed', 'barked', 'moved', 'complained', 'listened', 'roared', 'worked', 'apologized', 'escaped', 'slept', 'started-trending', 'fought']))

transloc_verbs = list(set(['arrived', 'skipped', 'went', 'staggered', 'sent', 'talked', 'took-off', 'marched', 'participated']))




obj_nouns = animate_nouns + inanimate_nouns 

all_nouns = animate_nouns + inanimate_nouns + location_nouns + other_nouns


all_vocab = all_nouns + determiners + adjectives + adverbs + aux + prepositions + DP + ambig_verbs + other_verbs + intrans_verbs + transloc_verbs


lexical_chunks_ep = {}
lexical_chunks_wd = {}

for noun in all_nouns:
	lexical_chunks_ep[noun] = {'syntax': ['NP', 'NP_ProgP', 'NP_VoiceP', 'NP_CP', 'NP_CP_null']}

	lexical_chunks_wd[noun] = {'syntax': ['NP', 'NP_CP', 'NP_CP_null']}


for verb in ambig_verbs:
	lexical_chunks_ep[verb] = { 'syntax' : ['Vt_act', 'Vt_pass']}
	lexical_chunks_wd[verb] = { 'syntax' : ['Vt_act', 'Vt_pass']}

for verb in other_verbs:
	lexical_chunks_ep[verb] = { 'syntax' : ['Vt_act']}
	lexical_chunks_wd[verb] = { 'syntax' : ['Vt_act']}


for verb in intrans_verbs:
	lexical_chunks_ep[verb] = { 'syntax' : ['V_intrans']}
	lexical_chunks_wd[verb] = { 'syntax' : ['V_intrans']}

for verb in transloc_verbs:
	lexical_chunks_ep[verb] = { 'syntax' : ['Vt_loc']}
	lexical_chunks_wd[verb] = { 'syntax' : ['Vt_loc']}


for det in determiners:
	lexical_chunks_ep[det] = { 'syntax' : ['Det']}
	lexical_chunks_wd[det] = { 'syntax' : ['Det']}


for prep in prepositions:
	lexical_chunks_ep[prep] = { 'syntax' : ['Prep']}
	lexical_chunks_wd[prep] = { 'syntax' : ['Prep']}

for adj in adjectives:
	lexical_chunks_ep[adj] = { 'syntax' : ['Adj']}
	lexical_chunks_wd[adj] = { 'syntax' : ['Adj']}

for adv in adverbs:
	lexical_chunks_ep[adv] = { 'syntax' : ['Adv']}
	lexical_chunks_wd[adv] = { 'syntax' : ['Adv']}


for a in aux:
	lexical_chunks_ep[a] = { 'syntax' : ['aux']}
	lexical_chunks_wd[a] = { 'syntax' : ['aux']}


for d in DP:
	lexical_chunks_ep[d] = { 'syntax' : ['DP']}
	lexical_chunks_wd[d] = { 'syntax' : ['DP']}

lexical_chunks_ep['who'] = {'syntax' : ['compsubj', 'compobj']}
lexical_chunks_wd['who'] = {'syntax' : ['compsubj', 'compobj']}

lexical_chunks_ep['comp_del'] = {'syntax' : ['compsubj', 'compobj']}
lexical_chunks_wd['comp_del'] = {'syntax' : ['compsubj', 'compobj']}

lexical_chunks_ep['was'] = {'syntax' : ['aux', 'aux_pass', 'aux_prog']}
lexical_chunks_wd['was'] = {'syntax' : ['aux', 'aux_pass', 'aux_prog']}

lexical_chunks_wd['aux_del'] = {'syntax' : ['aux', 'aux_pass', 'aux_prog']}



lexical_chunks_ep['being'] = {'syntax' : ['ProgP']}
lexical_chunks_wd['being'] = {'syntax' : ['ProgP']}

lexical_chunks_ep['and'] = {'syntax' : ['conj']}
lexical_chunks_wd['and'] = {'syntax' : ['conj']}

lexical_chunks_ep['.'] = {'syntax' : ['eos']}
lexical_chunks_wd['.'] = {'syntax' : ['eos']}



common_syntax_chunks = {
	'Det' : {
		'left': 'DP',
		'right': 'NP',
		'combinator': '/'
	},

	'DP' : {
		'left': 'DP',
		'right': '',
		'combinator': ''
	},

	'NP' : {
		'left': 'NP',
		'right': '',
		'combinator': ''
	},


	'NP_CP' : {
		'left': 'NP',
		'right': 'CP',
		'combinator': '/'
	},

	'NP_CP_null' : {
		'left': 'NP',
		'right': 'CP_null',
		'combinator': '/'
	},

	'Prep' : {
		'left': 'PP',
		'right': 'DP',
		'combinator': '/'
	},

	'Vt_pass': {
		'left': 'VoiceP',
		'right': 'PP',
		'combinator': '/'
	},

	'Vt_act': {
		'left': '(TP\\DP)',
		'right': 'DP',
		'combinator': '/'
	},

	'Vt_loc': {
		'left': '(TP\\DP)',
		'right': 'PP',
		'combinator': '/'
	},

	'V_intrans': {
		'left': 'TP',
		'right': 'DP',
		'combinator': '\\'
	},

	'compsubj': {
		'left': 'CP',
		'right': '(TP\\DP)',
		'combinator': '/'
	},

	'compobj': {
		'left': 'CP',
		'right': '(((TP\\DP)/DP)/DP)',
		'combinator': '/'
	},

	'compobj_null': {
		'left': 'CP_null',
		'right': '(((TP\\DP)/DP)/DP)',
		'combinator': '/'
	},

	'ProgP' : {
		'left': 'ProgP',
		'right': '',
		'combinator': ''
	},

	'aux': {
		'left': '(TP\\DP)',
		'right': '(NP/NP)',
		'combinator': '/'
	},

	'aux_pass': {
		'left': '(TP\\DP)',
		'right': 'VoiceP',
		'combinator': '/'
	},

	'aux_prog': {
		'left': '(TP\\DP)',
		'right': '(VoiceP/ProgP)',
		'combinator': '/'
	},

	'Adj': {
		'left': 'NP',
		'right': 'NP',
		'combinator': '/',
	},

	'Adv': {
		'left': 'TP',
		'right': 'TP',
		'combinator': '\\',
	},

	'conj': {
		'left': ' (TP/(TP\\DP))',
		'right': 'TP',
		'combinator': '\\',
	},

	'eos': {
		'left': 'end',
		'right': 'TP',
		'combinator': '\\',
	}
}

syntax_chunks_ep = {key:val for key, val in common_syntax_chunks.items()}


syntax_chunks_wd = {key:val for key, val in common_syntax_chunks.items()}


syntax_chunks_ep['NP_VoiceP'] = {
								'left': 'NP',
								'right': 'VoiceP',
								'combinator': '/'
								}


syntax_chunks_ep['NP_ProgP'] = {
								'left': 'NP',
								'right': '(VoiceP/ProgP)',
								'combinator': '/'
								}



syntax_chunks_wd['compsubj_null'] = {
								'left': 'CP_null',
								'right': '(TP\\DP)',
								'combinator': '/'
								}



with open('./data/rrcs.txt', 'r') as f:
	stims = f.readlines()

stims = [x.lower() for x in stims]
stims = [x.strip() for x in stims]

stims = ' '.join(stims)

stims = stims.split()

stim_vocab = set(stims)


missing_words = [x for x in stim_vocab if x not in all_vocab]

print(missing_words)
with open('./declmem/lexical_chunks_ep.pkl', 'wb') as f:
		pickle.dump(lexical_chunks_ep, f)


with open('./declmem/lexical_chunks_wd.pkl', 'wb') as f:
		pickle.dump(lexical_chunks_wd, f)


with open('./declmem/syntax_chunks_ep.pkl', 'wb') as f:
		pickle.dump(syntax_chunks_ep, f)


with open('./declmem/syntax_chunks_wd.pkl', 'wb') as f:
		pickle.dump(syntax_chunks_wd, f)
