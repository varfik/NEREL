import torch
from torch.optim import AdamW 
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os
import json
from collections import defaultdict

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=5, num_rel_labels=3):
        super().__init__()

        self.num_ner_labels = num_ner_labels 
        self.num_rel_labels = num_rel_labels 
        
        self.bert = AutoModel.from_pretrained(model_name)
        # self.config = AutoConfig.from_pretrained(model_name)
        
        # Enhanced NER Head
        self.ner_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ner_labels)  # Now 5 classes: O, B-PER, I-PER, B-PROF, I-PROF
        )
        
        # Relation Head remains the same
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_rel_labels)
        )

    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        ner_logits = self.ner_classifier(sequence_output)
        total_loss = 0
        rel_logits = None

        # NER loss
        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = attention_mask.view(-1) == 1
            ner_loss = loss_fct(
                ner_logits.view(-1, self.num_ner_labels)[active_loss],
                ner_labels.view(-1)[active_loss]
            )
            total_loss += ner_loss

        # Relation processing
        if rel_data and any(len(sample['pairs']) > 0 for sample in rel_data):
            rel_features, rel_labels = self._prepare_relation_features(sequence_output, rel_data)
            if rel_features is not None:
                rel_logits = self.rel_classifier(rel_features)
                
                if rel_labels:
                    rel_loss = nn.CrossEntropyLoss()(
                        rel_logits, 
                        torch.tensor(rel_labels, device=input_ids.device)
                    )
                    total_loss += rel_loss

        return {
            'ner_logits': ner_logits,
            'rel_logits': rel_logits,
            'loss': total_loss if total_loss != 0 else None
        }
    
    def _prepare_relation_features(self, sequence_output, rel_data):
        features, labels = [], []
        
        for batch_idx, sample in enumerate(rel_data):
            if not sample.get('pairs', []):
                continue
                
            # Get only valid entities that were properly tokenized
            valid_entities = [e for e in sample['entities'] if e['start'] <= e['end']]
            if len(valid_entities) < 2:
                continue
                
            # Create entity embeddings only for valid entities
            entity_embeddings = []
            for e in valid_entities:
                # Ensure we don't go beyond sequence length
                start = min(e['start'], sequence_output.size(1)-1)
                end = min(e['end'], sequence_output.size(1)-1)
                entity_embed = sequence_output[batch_idx, start:end+1].mean(dim=0)
                entity_embeddings.append(entity_embed)

            # Process relations - use direct indices since we've filtered entities
            for (e1_idx, e2_idx), label in zip(sample['pairs'], sample['labels']):
                # Check if indices are within bounds of our valid entities
                if e1_idx < len(valid_entities) and e2_idx < len(valid_entities):
                    feature = torch.cat([
                        entity_embeddings[e1_idx],
                        entity_embeddings[e2_idx],
                        sequence_output[batch_idx].mean(dim=0)  # context
                    ], dim=-1)
                    features.append(feature)
                    labels.append(label)
        
        return torch.stack(features) if features else None, labels

class NERELDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for txt_file in [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]:
            ann_path = os.path.join(self.data_dir, txt_file.replace('.txt', '.ann'))
            if not os.path.exists(ann_path):
                continue
                
            with open(os.path.join(self.data_dir, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
            
            entities, relations = self._parse_ann_file(ann_path)
            samples.append({'text': text, 'entities': entities, 'relations': relations})
        
        return samples
    
    def _parse_ann_file(self, ann_path):
        entities, relations = [], []
        entity_map = {}
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('T'):
                    parts = line.strip().split('\t')
                    entity_id = parts[0]
                    type_and_span = parts[1].split()
                    entity_type = type_and_span[0]
                    
                    if entity_type in ['PERSON', 'PROFESSION']:
                        entity = {
                            'id': entity_id,
                            'type': entity_type,
                            'start': int(type_and_span[1]),
                            'end': int(type_and_span[-1]),
                            'text': parts[2]
                        }
                        entities.append(entity)
                        entity_map[entity_id] = entity
                
                elif line.startswith('R'):
                    print("HEREEEEEEEEEEEEEEEEE")
                    parts = line.strip().split('\t')
                    rel_type, arg1, arg2 = parts[1].split()
                    arg1 = arg1.split(':')[1]
                    arg2 = arg2.split(':')[1]
                    
                    if rel_type in ['WORKS_AS', 'WORKPLACE'] and arg1 in entity_map and arg2 in entity_map:
                        relations.append({
                            'type': rel_type,
                            'arg1': arg1,
                            'arg2': arg2
                        })
        print(f"RELATIONS: {relations}")
        return entities, relations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True
        )
         # Initialize all labels as 'O' (0)
        ner_labels = [0] * len(encoding['input_ids'])
        
        # Align entities with tokens
        token_entities = []
        for orig_idx, entity in enumerate(sample['entities']):
            start_token = end_token = None
            for i, (start, end) in enumerate(encoding['offset_mapping']):
                if start <= entity['start'] < end and start_token is None:
                    start_token = i
                if start < entity['end'] <= end and end_token is None:
                    end_token = i
            
            if start_token is not None and end_token is not None:
                # Mark first token as B-ENTITY
                ner_labels[start_token] = 1 if entity['type'] == 'PERSON' else 3
                # Mark subsequent tokens as I-ENTITY
                for i in range(start_token+1, end_token+1):
                    ner_labels[i] = 2 if entity['type'] == 'PERSON' else 4
                
                token_entities.append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type'],
                    'original_idx': orig_idx  # Store original index for relation mapping
                })
        
        # Prepare relation data
        rel_data = {
            'entities': token_entities,
            'pairs': [],
            'labels': []
        }
        
        # Create mapping from original entity IDs to token entity indices
        id_to_idx = {e['id']: i for i, e in enumerate(sample['entities'])}
        token_entity_indices = {e['original_idx']: i for i, e in enumerate(token_entities)}
        
        for relation in sample['relations']:
            arg1_orig = id_to_idx.get(relation['arg1'], -1)
            arg2_orig = id_to_idx.get(relation['arg2'], -1)
            
            arg1_token = token_entity_indices.get(arg1_orig, -1)
            arg2_token = token_entity_indices.get(arg2_orig, -1)
            
            if arg1_token != -1 and arg2_token != -1:
                rel_data['pairs'].append((arg1_token, arg2_token))
                rel_data['labels'].append(0 if relation['type'] == 'WORKS_AS' else 1)
        
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'ner_labels': torch.tensor(ner_labels),
            'rel_data': rel_data
        }

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    
    padded_batch = {
        'input_ids': torch.stack([
            torch.nn.functional.pad(
                item['input_ids'],
                (0, max_len - len(item['input_ids'])),
                value=0
            ) for item in batch
        ]),
        'attention_mask': torch.stack([
            torch.nn.functional.pad(
                item['attention_mask'],
                (0, max_len - len(item['attention_mask'])),
                value=0
            ) for item in batch
        ]),
        'ner_labels': torch.stack([
            torch.nn.functional.pad(
                item['ner_labels'],
                (0, max_len - len(item['ner_labels'])),
                value=0
            ) for item in batch
        ]),
        'rel_data': [item['rel_data'] for item in batch]
    }
    return padded_batch

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = NERRelationModel().to(device)
    
    train_dataset = NERELDataset("NEREL/NEREL-v1.1/train", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(3):
        model.train()
        epoch_loss = ner_correct = ner_total = rel_correct = rel_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'ner_labels': batch['ner_labels'].to(device),
                'rel_data': batch['rel_data']
            }
            
            outputs = model(**inputs)
            outputs['loss'].backward()
            optimizer.step()
            
            epoch_loss += outputs['loss'].item()
            
            # Calculate NER accuracy
            ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)
            active_mask = inputs['attention_mask'] == 1
            ner_correct += ((ner_preds == inputs['ner_labels']) & active_mask).sum().item()
            ner_total += active_mask.sum().item()
            
            # Calculate relation accuracy
            if outputs['rel_logits'] is not None:
                rel_preds = torch.argmax(outputs['rel_logits'], dim=-1)
                rel_labels = [
                    label for sample in batch['rel_data'] 
                    for label in sample.get('labels', [])
                ]
                
                if rel_labels:
                    rel_correct += (rel_preds == torch.tensor(rel_labels, device=device)).sum().item()
                    rel_total += len(rel_labels)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"NER Accuracy: {ner_correct/ner_total:.2%}")
        if rel_total > 0:
            print(f"Relation Accuracy: {rel_correct/rel_total:.2%}")
    
    return model, tokenizer

def predict(text, model, tokenizer, device="cuda"):
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(device)
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
    
    # Decode NER predictions
    ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, ner_preds)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if pred != 0:  # Not 'O'
            entity_type = "PERSON" if pred == 1 else "PROFESSION"
            
            if current_entity and current_entity['type'] == entity_type:
                current_entity['end'] = i
                current_entity['text'] += token.replace('##', '')
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'text': token.replace('##', '')
                }
    
    if current_entity:
        entities.append(current_entity)
    
    # Extract relations
    relations = []
    if len(entities) >= 2 and outputs['rel_logits'] is not None:
        sequence_output = model.bert(
            encoding['input_ids'], 
            encoding['attention_mask']
        ).last_hidden_state
        
        context = sequence_output.mean(dim=1)
        pairs = [(i, j) for i in range(len(entities)) for j in range(len(entities)) if i != j]
        
        for i, j in pairs:
            e1 = sequence_output[:, entities[i]['start']:entities[i]['end']+1].mean(dim=1)
            e2 = sequence_output[:, entities[j]['start']:entities[j]['end']+1].mean(dim=1)
            feature = torch.cat([e1, e2, context], dim=-1)
            
            with torch.no_grad():
                rel_prob = torch.softmax(model.rel_classifier(feature), dim=-1)[0]
            
            if rel_prob.argmax() != 2:  # Not 'NO_RELATION'
                relations.append({
                    'type': "WORKS_AS" if rel_prob.argmax() == 0 else "WORKPLACE",
                    'arg1': entities[i],
                    'arg2': entities[j],
                    'confidence': rel_prob.max().item()
                })
    
    return {
        'text': text,
        'entities': entities,
        'relations': relations
    }

# Usage example
if __name__ == "__main__":
    model, tokenizer = train_model()
    
    text = "Айрат Мурзагалиев, заместитель начальника управления президента РФ, встретился с главой администрации Уфы."
    result = predict(text, model, tokenizer)
    
    print("\nEntities:")
    for e in result['entities']:
        print(f"{e['type']}: {e['text']}")
    
    print("\nRelations:")
    for r in result['relations']:
        print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (conf: {r['confidence']:.2f})")
