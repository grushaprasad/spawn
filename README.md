# Serial Parsing in ACT-R With Null elements

This code implements an ACT-R based serial parser with re-analysis using a CCG inspired grammar formalism. The code can also be used to generate priming predictions for a comprehension-to-production priming paradigm. 

## General components of the code
In most cases, applying SPAWN to other phenomena will *not* require modifying the following files:
### model.py
The model class with ACT-R based methods for memory retrieval and activation updates.

### supertagger.py
This includes the algorithsm for parsing, supertagging + combining, and re-analysis. 

### ccg.py
Implementation of CCG rule application with the six CCG application rules + type raising. 

## Experiment-specific components of the code
The following files contain code for the specific set of experiments described in chapter 2 of [this dissertation](https://jscholarship.library.jhu.edu/bitstream/handle/1774.2/67607/SAIPRASAD-DISSERTATION-2022.pdf?sequence=1). In most cases, applying SPAWN to other phenomena will require modifying these files.

### create_declmem.py
Specifying the declarative memory (i.e. "grammar" and "vocabulary"). 

### create_training_dat.py
Templates to create data to train SPAWN models.  

### train.py
Code to train models on a pre-tokenized corpus where tokens are separated by space. All model hyperparameters are specified in this file. 

### prime.py
Code to evaluate and "adapt" or "fine-tune" already trained models.






