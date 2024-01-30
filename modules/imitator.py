from datasets import Dataset
import pandas as pd
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

ds = Dataset.from_file("../data/medline_data.arrow")

expository_text_corpus = []

for row in ds:
    expository_text_corpus.append(row['output_aug'])

pd.DataFrame(expository_text_corpus).to_csv("../data/expository_text_corpus.csv", index = False)
text_data = open('../data/expository_text_corpus.txt', 'w')

for row in ds:
  text = row["output_aug"]
  text_data.write(text)

text_data.close()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
train_dataset = TextDataset(tokenizer = tokenizer, file_path = '../data/expository_text_corpus.txt', block_size = 128)
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)

training_args = TrainingArguments(
        output_dir = 'imitator_training_args/',
        overwrite_output_dir = False,
        per_device_train_batch_size = 8,
        num_train_epochs = 5.0
)

trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset
)

trainer.train()
trainer.save_model('imitator_model/')