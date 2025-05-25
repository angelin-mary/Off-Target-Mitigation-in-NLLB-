import json
from datasets import load_dataset, Dataset
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
)
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def preprocess_data():
    print("Loading the FLORES-200 dataset...")
    dataset = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)

    print("Preparing data for language identification...")
    processed_data = []
    for example in tqdm(dataset, desc="Processing examples"):
        for lang_code, sentence in example.items():
            if lang_code.startswith('sentence_') and sentence:
                lang_code_clean = lang_code.replace('sentence_', '')
                processed_data.append({
                    "text": sentence.strip(),
                    "label": lang_code_clean
                })

    print(f"Extracted {len(processed_data)} sentence-label pairs.")

    print("Splitting data into training, validation, and test sets...")
    train_data, temp_data = train_test_split(
        processed_data, test_size=0.2, random_state=42,
        stratify=[item['label'] for item in processed_data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[item['label'] for item in temp_data]
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    print("Loading the tokenizer from the checkpoint...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("path/to/your/checkpoint")  # <- Replace with your model checkpoint path

    print("Tokenizing the datasets...")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

    train_tokenized = train_dataset.map(preprocess_function, batched=True, desc="Tokenizing training data")
    val_tokenized = val_dataset.map(preprocess_function, batched=True, desc="Tokenizing validation data")
    test_tokenized = test_dataset.map(preprocess_function, batched=True, desc="Tokenizing test data")

    # Encode string labels to integers
    print("Encoding labels...")
    labels = sorted(set([item['label'] for item in train_data]))
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    def encode_labels(examples):
        examples["label"] = label_to_id[examples["label"]]
        return examples

    train_tokenized = train_tokenized.map(encode_labels, desc="Encoding training labels")
    val_tokenized = val_tokenized.map(encode_labels, desc="Encoding validation labels")
    test_tokenized = test_tokenized.map(encode_labels, desc="Encoding test labels")

    print("Setting dataset format to PyTorch tensors...")
    test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return test_tokenized, tokenizer, label_to_id, id_to_label

def evaluate_model(test_tokenized, tokenizer, label_to_id, id_to_label):
    print("Loading the trained model from the checkpoint...")
    model = XLMRobertaForSequenceClassification.from_pretrained("path/to/your/checkpoint") # <- Replace with your model checkpoint path
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    print("Creating DataLoader for the test set...")
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=16)

    all_predictions = []
    all_labels = []

    print("Evaluating the model on the test set...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print("\nCalculating evaluation metrics...")
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    print(f"\nEvaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nGenerating classification report...")
    target_names = [id_to_label[i] for i in range(len(id_to_label))]
    print(classification_report(all_labels, all_predictions, target_names=target_names))

def main():
    print("Starting preprocessing...")
    test_tokenized, tokenizer, label_to_id, id_to_label = preprocess_data()
    print("Preprocessing completed.\n")

    evaluate_model(test_tokenized, tokenizer, label_to_id, id_to_label)
    print("\nEvaluation completed.")

if __name__ == "__main__":
    main()
