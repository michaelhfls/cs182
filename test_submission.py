import json, sys
import pandas as pd


from simpletransformers.classification import ClassificationModel

def eval(text):
	# This is where you call your model to get the number of stars output
	# The line below should match the exact line used in the .ipynb when the model was created, including changed hyperparameters
	# except for the second argument which specifies whether to make a new model or load one from a location (e.g. outputs/)
	model = ClassificationModel('bert', 'outputs/', num_labels=6, use_cuda=False)
	predictions, raw_outputs = model.predict([text])
	return predictions[0]

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")
