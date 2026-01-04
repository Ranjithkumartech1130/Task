# Transformers example - fake sentiment classification
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Fake text dataset
texts = [
    "I love this product", "This is amazing", "Worst experience ever",
    "I hate it", "Absolutely fantastic", "Not good", "I am happy",
    "Very bad", "Excellent service", "Terrible quality"
] * 10  # 100 samples

labels = [1,1,0,0,1,0,1,0,1,0] * 10  # 0 = negative, 1 = positive

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 3: Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=32)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=32)

# Step 4: Create torch dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, y_train)
test_dataset = SentimentDataset(test_encodings, y_test)

# Step 5: Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to="none"
)

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 8: Train
trainer.train()

# Step 9: Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
