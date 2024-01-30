from transformers import AutoModelForSeq2SeqLM, GPT2LMHeadModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import Dataset
import numpy as np
import nltk
import random

nltk.download('punkt')

# Load in medline dataset
train_data = Dataset.from_file('./data/train/medline_train_data.arrow')
test_data = Dataset.from_file('./data/test/medline_test_data.arrow')
print(test_data)

# Load in pretrained models
imitator = GPT2LMHeadModel.from_pretrained('./modules/imitator_model').to('cuda')
retriever = AutoModelForSequenceClassification.from_pretrained('./modules/retriever_model').to('cuda')
paraphraser = AutoModelForSeq2SeqLM.from_pretrained('./modules/paraphraser_model').to('cuda')

#Load in tokenizers
imitator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
retriever_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
paraphraser_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

special_tokens = {'additional_special_tokens': ['<|topic|>', '<|style|>', '<|fact|>']}
num_added_toks = paraphraser_tokenizer.add_special_tokens(special_tokens)
paraphraser.resize_token_embeddings(len(paraphraser_tokenizer))

# Function for loading in prefix input r
def get_prefix(topic):
  return f'{topic} is used to treat'

# Function for generating next context sentence plan in output
def get_imitated_sentence(prefix):
  inputs = imitator_tokenizer(prefix, return_tensors = 'pt').to('cuda')
  generation_output = imitator.generate(**inputs, return_dict_in_generate = True, output_scores = True, max_new_tokens = 64)
  decoded_output = imitator_tokenizer.decode(generation_output.sequences[0])
  new_sentence = decoded_output.split(prefix)[1].split('.')[0] + '.'
  return new_sentence

# Function for retrieving top k facts based off of current context sentence plan
def get_retrieved_facts(sentence, fact_embeddings, k):
  tokenized_sentence = retriever_tokenizer(sentence, return_tensors = 'pt', truncation = True).input_ids.to('cuda')
  sentence_embedding = retriever(tokenized_sentence, output_hidden_states = True).hidden_states[-1].mean(axis = 1).to('cpu').detach()
  fact_scores = []

  for fact_embedding in fact_embeddings:
    fact_scores.append([float((fact_embedding[0] @ sentence_embedding.T)[0][0]), fact_embedding[1]])

  top_k_facts = sorted(fact_scores, key = lambda x: x[0], reverse = True)[:k]
  return [fact[1] for fact in top_k_facts]

# Function for combining current context sentence plan with top k facts for topic t
def get_paraphrased_sentence(sentence, facts, topic):
  facts_combined = ''
  for fact in facts:
    facts_combined = facts_combined + fact

  model_input = ['<|topic|>' + topic + ' <|fact|> ' + facts_combined + ' <|style|> ' + sentence]
  model_input = paraphraser_tokenizer(model_input, max_length = 512, truncation = True, return_tensors='pt')
  attention_mask = model_input.attention_mask.to('cuda')
  input_ids = model_input.input_ids.to('cuda')
  outputs = paraphraser.generate(input_ids, attention_mask = attention_mask, max_length = 512).to('cpu').detach()
  output_str = paraphraser_tokenizer.batch_decode(outputs, skip_special_tokens = True)
  return (''.join(output_str).split('.')[0] + '.')[len(topic) + 1:]

# Function for generating expository text using IRP
def IRP(topic, prefix, facts):
  output = prefix

  fact_embeddings = []
  for fact in facts:
    tokenized_fact = retriever_tokenizer(fact, return_tensors = 'pt', truncation = True).input_ids.to('cuda')
    fact_embedding = retriever(tokenized_fact, output_hidden_states = True).hidden_states[-1].mean(axis = 1).to('cpu').detach()
    fact_embeddings.append([fact_embedding, fact])

  for i in range(1000):
    # Get new imitated sentence
    imitated_sentence = get_imitated_sentence(output)

    if i == 0:
      imitated_sentence = prefix + imitated_sentence

    if imitated_sentence == '<|endoftext|>':
      break

    # Retrieve top facts
    retrieved_facts = get_retrieved_facts(imitated_sentence, fact_embeddings, 1)

    # Paraphrase imitated sentence with retrieved facts
    paraphrased_sentence = get_paraphrased_sentence(imitated_sentence, retrieved_facts, topic)

    if i == 0:
      output = paraphrased_sentence.lstrip().rstrip()
    else:
      output = (output + ' ' + paraphrased_sentence).lstrip().rstrip()

    if i == 3:
      break

  return output

# Inference loop over dataset
results = []
random_indices = random.sample(range(len(test_data)), 5)

for idx in random_indices:
  row = test_data[idx]
  topic = row['title']
  prefix = get_prefix(topic)
  facts = row['facts']
  expository_text = IRP(topic, prefix, facts)
  results.append({'topic': topic, 'prefix': prefix, 'facts': facts, 'expository_text': expository_text})