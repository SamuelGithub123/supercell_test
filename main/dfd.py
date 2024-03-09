import pandas as pd
import torch
import torch.nn.functional as F 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification,Trainer, TrainingArguments
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
df_chat_messages_1 = pd.read_csv('../data/chat_messages_1.csv',low_memory=False)
texts_column = df_chat_messages_1['raw_message'].head(100)  
risk_column = df_chat_messages_1['risk'].head(100)    
processed_risk = risk_column.apply(lambda x: 0 if float(x) <= 3 else 1)


texts_column_list = texts_column.tolist()
processed_risk_list = processed_risk.tolist()


train_texts, val_texts, train_labels, val_labels = train_test_split(texts_column_list, processed_risk_list, test_size=.2)



task='offensive'
MODEL = f"../twitter-roberta-base-{task}"

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset        # training dataset
       # evaluation dataset
)

trainer.train()