import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BertConfig, 
    BertEncoder
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split

model_name = "facebook/nllb-200-distilled-600M"
source_languages = ["eng_Latn", "spa_Latn", "mal_Mlym"]
target_languages = [
    "arb_Latn", "sat_Olck", "taq_Tfng", "min_Arab", "acm_Arab", "ars_Arab", "acq_Arab", "prs_Arab",
    "aka_Latn", "ary_Arab", "ajp_Arab", "dyu_Latn", "apc_Arab", "aeb_Arab", "arz_Arab", "kmb_Latn",
    "zho_Hant", "hrv_Latn", "awa_Deva", "bod_Tibt", "kin_Latn", "bjn_Arab", "knc_Arab"
]
lang2id = {lang: idx for idx, lang in enumerate(target_languages)}
alpha = 0.2 

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class TLPSeq2SeqModel(nn.Module):
    def __init__(self, base_model, n_langs):
        super().__init__()
        self.base_model = base_model
        self.n_langs = n_langs

        
        tlp_config = BertConfig(
            hidden_size=base_model.config.d_model,
            num_hidden_layers=2,
            num_attention_heads=8,  
            intermediate_size=4 * base_model.config.d_model,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        self.tlp_encoder = BertEncoder(tlp_config)
        self.lang_classifier = nn.Linear(base_model.config.d_model, n_langs)


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        loss_nmt = outputs.loss
        last_states = outputs.decoder_hidden_states[-1]  
        pooled = last_states.mean(dim=1)                
        lang_logits = self.lang_classifier(pooled)
        return outputs.logits, lang_logits, loss_nmt

model = TLPSeq2SeqModel(base_model, n_langs=len(target_languages))

raw = load_dataset("facebook/flores", "all", trust_remote_code=True)
examples = []
for item in raw['devtest']:
    for src_lang in source_languages:
        src_text = item.get(f'sentence_{src_lang}')
        if not src_text:
            continue
        for tgt_lang in target_languages:
            tgt_text = item.get(f'sentence_{tgt_lang}')
            if not tgt_text:
                continue
            examples.append({
                'src': f'>>{tgt_lang}<< {src_text}',
                'tgt': tgt_text,
                'lang_id': lang2id[tgt_lang]
            })

train_val, test = train_test_split(examples, test_size=0.1, random_state=42)
train, val   = train_test_split(train_val, test_size=0.1, random_state=42)

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item['src'], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        dec = self.tokenizer(
            item['tgt'], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        labels = dec.input_ids.squeeze(0).clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels': labels,
            'lang_id': torch.tensor(item['lang_id'], dtype=torch.long)
        }

train_dataset = TranslationDataset(train, tokenizer)
val_dataset   = TranslationDataset(val,   tokenizer)
test_dataset  = TranslationDataset(test,  tokenizer)

def collate_fn(batch):
    input_ids      = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels         = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
    lang_id        = torch.stack([b['lang_id'] for b in batch])
    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
        'lang_id':        lang_id
    }

class TLPTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits, lang_logits, loss_nmt = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs.get('labels')
        )
        loss_tlp = F.cross_entropy(lang_logits, inputs['lang_id'])
        loss = (1 - alpha) * loss_nmt + alpha * loss_tlp
        return (loss, (logits, lang_logits)) if return_outputs else loss

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

training_args = Seq2SeqTrainingArguments(
    output_dir="path/to/save/nllb_tlp_model",  # <- Replace with your desired save path
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="no",
    load_best_model_at_end=False
)

trainer = TLPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

eval_metrics = trainer.evaluate(test_dataset)
print("Test results:", eval_metrics)

output_dir = "./nllb_tlp_model"
os.makedirs(output_dir, exist_ok=True)

trainer.model.base_model.config.save_pretrained(output_dir)
torch.save(
    trainer.model.base_model.state_dict(),
    os.path.join(output_dir, "pytorch_model.bin")
)

torch.save(
    trainer.model.lang_classifier.state_dict(),
    os.path.join(output_dir, "lang_classifier.pt")
)

tokenizer.save_pretrained(output_dir)

print(f"Saved model, TLP head, and tokenizer to {output_dir}")
