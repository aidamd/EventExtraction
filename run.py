from preprocess import GenerateTrain
from MICNN import *
from MILSTM import *
import json
import pickle
# Phase one is a multi instance learner that find the hate crime articles

# Get the training set from Propublica and Patch
if os.path.isfile("data.pkl"):
    data = pickle.load(open("data.pkl", "rb"))
else:
    data = GenerateTrain(2000)

train_batches, dev_batches, vocabs = data
# Get params
params = json.load(open("params.json", "r"))

if params["model"] == "CNN":
    cnn = MICNN(params, vocabs)
    cnn.build()
    cnn.run_model(train_batches, dev_batches)
else:
    lstm = MILSTM(params, vocabs)
    lstm.build()
    lstm.run_model(train_batches, dev_batches)
