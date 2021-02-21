import pickle
import json

with open('./chatbot_dict.bin', "rb") as f:
    bin = pickle.load(f)

with open('./chatbot_dict.json', 'rb') as f:
    json = json.load(f)

print("json: {}, bin: {}".format(len(bin), len(json)))