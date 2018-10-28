# Singzy_assnmnt_nlp
Importing the usefull library 
import spacy #this library has has predefined model for NER
from spacy.util import minibatch, compounding
# minibatch will help to iterate over the batch of items
# yield an infinite series of compounding values
import plac
import random
# i took 8 diiferent sentance to train the data in which it contain the indian names
model=("Model name. Defaults to blank 'en' model.", "option", "m", str)
# since the problem statment was to recognige the only name wherre the Spacy Library has the defaolt model to recognize the person name like "en"
n_iter=("Number of training iterations", "option", "n", int)
#defining the convinient iteration number to trian the data
def main(model=None, n_iter=100):
"""Load the model, set up the pipeline and train the entity recognizer."""
if model is not None:
nlp = spacy.load(model)  # load existing spaCy model
else:
nlp = spacy.blank('en')  # if model not fit to existing model,create blank Language class
print("to Create blank 'en' model")
if 'ner' not in nlp.pipe_names: # since we need only the person name and it is is prasent in the 'ner'
ner = nlp.create_pipe('ner')
nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
ner = nlp.get_pipe('ner')
# adding the lable so it can identify the entities through this mention belove loop
for _, annotations in train_data:
for ent in annotations.get('entities'):
ner.add_label(ent[2])
#get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes): 
 # since we are using the exixting model so need to deseble all other pipeline component during training
 optimizer = nlp.begin_training()# optimizing the training data for update in the ner
 for itn in range(n_iter): # taking each iteration on the training data
 random.shuffle(train_data)
 losses = {}
 # batch up the examples using spaCy's minibatch
 batches = minibatch(train_data, size=compounding(4., 32., 1.001))
 for batch in batches:
 texts, annotations = zip(*batch)
 nlp.update( texts, annotations, drop=0.5, sgd=optimizer,losses=losses)
 # texets=batch of texts
 #  annotations=batch of annotations
 # (drop=0.5) =dropout - make it harder to memorise data
 # (sgd=optimizer)=callable to update weight

 # test the trained model
  for text, _ in train_data:
  doc = nlp(text)
  print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        
if __name__ == '__main__':
    plac.call(main)
