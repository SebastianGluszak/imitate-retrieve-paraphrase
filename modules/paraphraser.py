from datasets import Dataset, load_dataset, DatasetDict
import nltk
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSequenceClassification
import torch
import tqdm

data = load_dataset('nbalepur/expository_documents_medicine')
train_data, test_data = data['train'], data['test']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
nltk.download('punkt')

model = AutoModelForSequenceClassification.from_pretrained('./retriever_model', num_labels = max([len(nltk.sent_tokenize(sent)) for sent in train_data['output_aug']])).to('cuda')

def get_embedding_from_class(sent):
    tok = tokenizer(sent, return_tensors='pt', truncation=True).input_ids.to('cuda')
    return model(tok, output_hidden_states=True).hidden_states[-1].mean(axis = 1).to('cpu').detach()

paraphrase_ds_train = {'title': [], 'facts': [], 'style': [], 'output': []}

all_output_texts = []
all_outputs = train_data['output_aug']
for text in all_outputs:
    all_output_texts.extend(nltk.sent_tokenize(text))
all_output_embs = torch.concat([get_embedding_from_class(s) for s in all_output_texts])

for data_num in tqdm.tqdm(range(len(train_data))):

    input_text = list(set(train_data['web_sentences_with_desc'][data_num]))
    info_embeddings = []

    for s in input_text:
        info_embeddings.append(get_embedding_from_class(s))
    info_embeddings = torch.concat(info_embeddings)

    for sent in nltk.sent_tokenize(train_data['output_aug'][data_num]):
        sent_emb = get_embedding_from_class(sent)
        style_sims = (all_output_embs @ sent_emb.T).squeeze(1)
        for idx in np.random.choice(torch.argsort(style_sims, descending=True)[1:50], 10):
            curr_style = all_output_texts[idx]
            style_emb = get_embedding_from_class(sent)
            fact_sims = (info_embeddings @ sent_emb.T).squeeze(1)

            facts = [input_text[idx_] for idx_ in torch.argsort(fact_sims, descending=True)[:10]]

            paraphrase_ds_train['facts'].append(facts)
            paraphrase_ds_train['style'].append(curr_style)
            paraphrase_ds_train['output'].append(sent)
            paraphrase_ds_train['title'].append(train_data['title'][data_num])

train_data = Dataset.from_dict(paraphrase_ds_train)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
nltk.download('punkt')

model = AutoModelForSequenceClassification.from_pretrained('./retriever_model', num_labels = max([len(nltk.sent_tokenize(sent)) for sent in test_data['output_aug']])).to('cuda')

def get_embedding_from_class(sent):
    tok = tokenizer(sent, return_tensors='pt', truncation=True).input_ids.to('cuda')
    return model(tok, output_hidden_states=True).hidden_states[-1].mean(axis = 1).to('cpu').detach()

paraphrase_ds_test = {'title': [], 'facts': [], 'style': [], 'output': []}

all_output_texts = []
all_outputs = test_data['output_aug']
for text in all_outputs:
    all_output_texts.extend(nltk.sent_tokenize(text))
all_output_embs = torch.concat([get_embedding_from_class(s) for s in all_output_texts])

for data_num in tqdm.tqdm(range(len(test_data))):

    input_text = list(set(test_data['web_sentences_with_desc'][data_num]))
    info_embeddings = []

    for s in input_text:
        info_embeddings.append(get_embedding_from_class(s))
    info_embeddings = torch.concat(info_embeddings)

    for sent in nltk.sent_tokenize(test_data['output_aug'][data_num]):
        sent_emb = get_embedding_from_class(sent)
        style_sims = (all_output_embs @ sent_emb.T).squeeze(1)
        for idx in np.random.choice(torch.argsort(style_sims, descending=True)[1:50], 10):
            curr_style = all_output_texts[idx]
            style_emb = get_embedding_from_class(sent)
            fact_sims = (info_embeddings @ sent_emb.T).squeeze(1)

            facts = [input_text[idx_] for idx_ in torch.argsort(fact_sims, descending=True)[:10]]

            paraphrase_ds_test['facts'].append(facts)
            paraphrase_ds_test['style'].append(curr_style)
            paraphrase_ds_test['output'].append(sent)
            paraphrase_ds_test['title'].append(test_data['title'][data_num])

test_data = Dataset.from_dict(paraphrase_ds_test)

new_ds = DatasetDict({'train': train_data, 'test': test_data})
new_ds.save_to_disk('../data')

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

special_tokens_dict = {'additional_special_tokens': ['<|topic|>', '<|style|>', '<|fact|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

batch_size = 16

training_args = Seq2SeqTrainingArguments(
    output_dir = './paraphraser_training_args',
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    weight_decay = 0.01,
    save_strategy = 'no',
    num_train_epochs = 5,
    predict_with_generate = True,
    gradient_accumulation_steps = 8,
    logging_steps = 100
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_data,
    data_collator = data_collator,
    tokenizer = tokenizer
)

trainer.save_model('./paraphraser_model')