from preprocess import GenerateTrain
from MICNN import *
from MILSTM import *
from Entity import Entity
import json
import pickle
# Phase one is a multi instance learner that find the hate crime articles

# Get the training set from Propublica and Patch
if os.path.isfile("data.pkl"):
    data = pickle.load(open("data.pkl", "rb"))
else:
    data = GenerateTrain(2000)

train_batches, dev_batches, vocabs = data
test_batches = pickle.load(open("patch.pkl", "rb"))
# Get params
params = json.load(open("params.json", "r"))

if params["task"] == "hate":
    if params["model"] == "CNN":
        cnn = MICNN(params, vocabs)
        cnn.build()
        predictions = cnn.run_model(train_batches, dev_batches, test_batches)
    else:
        lstm = MILSTM(params, vocabs)
        lstm.build()
        predictions = lstm.run_model(train_batches, dev_batches, test_batches)
else:
    train_batches = [train for train in train_batches if train[2] == 1]
    dev_batches = [dev for dev in dev_batches if dev[2] == 1]
    t_weights = [1 - (Counter([train[3] for train in train_batches])[i] / len(train_batches)) for i in range(6)]
    a_weights = [1 - (Counter([train[4] for train in train_batches])[i] / len(train_batches)) for i in range(5)]
    entity = Entity(params, vocabs)
    entity.build()
    entity.run_model(train_batches, dev_batches, test_batches, (t_weights, a_weights))
