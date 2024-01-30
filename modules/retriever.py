from datasets import Dataset
import numpy as np
import nltk
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification

ds = Dataset.from_file('../data/medline_data.arrow')

train_data = {'text': [], 'label': []}
nltk.download('punkt')

for row in ds:
    text = row['output_aug']
    sentences = nltk.sent_tokenize(text, language = 'english')

    for idx, sentence in enumerate(sentences):
        train_data['text'].append(sentence)
        train_data['label'].append(idx)

train_data = Dataset.from_dict(train_data)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_data = train_data.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = len(np.unique(train_data['label'])))

training_args = TrainingArguments(
    output_dir = 'retriever_training_args/',
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 1,
    weight_decay = 0.01,
    save_strategy = 'no'
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_data,
    tokenizer = tokenizer,
    data_collator = data_collator
)

trainer.train()
trainer.save_model('retriever_model/')