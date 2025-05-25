from sklearn.model_selection import train_test_split
import json
from datasets import load_dataset, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load FLORES-200 devtest set and restructure data as text-label pairs
dataset = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
processed_data = []

for example in dataset:
    for lang_code, sentence in example.items():
        if lang_code.startswith('sentence_'):
            lang_code = lang_code.replace('sentence_', '')
            processed_data.append({
                "text": sentence.strip(),
                "label": lang_code
            })

print(f"Extracted {len(processed_data)} sentence-label pairs.")

train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

with open("train_data.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
with open("val_data.json", 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)
with open("test_data.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

def print_samples(data, n=5):
    for sample in data[:n]:
        print(f"Text: {sample['text']}\nLabel: {sample['label']}\n{'-'*50}")

print("\n--- Training Samples ---")
print_samples(train_data)
print("\n--- Validation Samples ---")
print_samples(val_data)
print("\n--- Test Samples ---")
print_samples(test_data)

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

train_data = load_json_data("train_data.json")
val_data = load_json_data("val_data.json")
test_data = load_json_data("test_data.json")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

# Tokenize text data using XLM-R tokenizer
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

train_tokenized = train_dataset.map(preprocess_function, batched=True)
val_tokenized = val_dataset.map(preprocess_function, batched=True)
test_tokenized = test_dataset.map(preprocess_function, batched=True)

# Encode labels as integers
labels = sorted(set([item['label'] for item in train_data]))
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

def encode_labels(examples):
    examples["label"] = label_to_id[examples["label"]]
    return examples

train_tokenized = train_tokenized.map(encode_labels)
val_tokenized = val_tokenized.map(encode_labels)
test_tokenized = test_tokenized.map(encode_labels)

model = XLMRobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_to_id)
)

def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="path/to/save/language_identifier",  # <- Replace with your desired save path
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("Evaluating on validation data...")
validation_results = trainer.evaluate(eval_dataset=val_tokenized)
print(f"Validation results: {validation_results}")

print("Evaluating on test data...")
test_results = trainer.evaluate(eval_dataset=test_tokenized)
print(f"Test results: {test_results}")

trainer.save_model("language_identifier_model")

loaded_model = XLMRobertaForSequenceClassification.from_pretrained("language_identifier_model")
loaded_tokenizer = XLMRobertaTokenizer.from_pretrained("language_identifier_model")

def predict_language(sentence, model, tokenizer, label_map):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_map[prediction]

sample_sentences = [
    "Hello, how are you?",
    "Hola, ¿cómo estás?",
    "Bonjour, comment ça va?",
    "你好，你怎么样？"
]

for sentence in sample_sentences:
    predicted_language = predict_language(sentence, loaded_model, loaded_tokenizer, id_to_label)
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Language: {predicted_language}")
    print("-" * 50)
