# Transformers - Topic Classification (Fake Dataset)

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Step 1: Fake text dataset
texts = [
    "The team won the match",
    "The player scored a goal",
    "Election results were announced",
    "The minister gave a speech",
    "New smartphone launched today",
    "AI is changing the world",
] * 10   # 60 samples

labels = [
    0, 0,   # Sports
    1, 1,   # Politics
    2, 2    # Technology
] * 10

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Step 3: Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_enc = tokenizer(X_train, padding=True, truncation=True, max_length=32)
test_enc = tokenizer(X_test, padding=True, truncation=True, max_length=32)

# Step 4: Torch Dataset
class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TopicDataset(train_enc, y_train)
test_dataset = TopicDataset(test_enc, y_test)

# Step 5: Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# Step 6: Training Arguments
training_args = TrainingArguments(
    output_dir="./topic_results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 8: Train & Evaluate
trainer.train()
print(trainer.evaluate())
