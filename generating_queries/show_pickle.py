import numpy as np
import pickle

File2Open='training_queries_refine.pickle'

with open(File2Open, 'rb') as f:
    loaded_object = pickle.load(f)
    
print(loaded_object[0]["positives"])
