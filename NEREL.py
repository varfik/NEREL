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
        bert_hidden_size = self.bert.config.hidden_size
        
        # Enhanced NER Head
        self.ner_classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ner_labels)  # Now 5 classes: O, B-PER, I-PER, B-PROF, I-PROF
        )
        
        # Relation Head remains the same
        self.rel_classifier = nn.Sequential(
            nn.Linear(bert_hidden_size  * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_rel_labels)
        )
        print(f"Initialized model with bert_hidden_size={bert_hidden_size}")

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
                    # Get the embeddings for this pair
                    e1_embed = entity_embeddings[e1_idx]
                    e2_embed = entity_embeddings[e2_idx]
                    context = sequence_output[batch_idx].mean(dim=0)
                    
                    # Concatenate features
                    feature = torch.cat([e1_embed, e2_embed, context], dim=-1)
                    
                    # Check feature size matches classifier input
                    expected_size = self.bert.config.hidden_size * 3
                    if feature.size(0) != expected_size:
                        print(f"Warning: Feature size mismatch. Expected {expected_size}, got {feature.size(0)}")
                        continue
                    
                    features.append(feature)
                    labels.append(label)
        
        if features:
            return torch.stack(features), labels
        return None, labels

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

        print(f"\nOriginal text: {sample['text']}")
        print(f"Original entities: {sample['entities']}")
        print(f"Original relations: {sample['relations']}")
        
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True
        )

        # Выведем информацию о токенизации
        print("\nTokenization info:")
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        for i, (token, (start, end)) in enumerate(zip(tokens, encoding['offset_mapping'])):
            print(f"{i}: {token} (chars: {start}-{end})")
        
         # Initialize all labels as 'O' (0)
        ner_labels = [0] * len(encoding['input_ids'])
        
        # Create mapping from entity ID to entity index in original sample
        entity_id_to_idx = {e['id']: idx for idx, e in enumerate(sample['entities'])}
        
        # Align entities with tokens
        token_entities = []
        for entity in sample['entities']:
            start_token = end_token = None
            for i, (start, end) in enumerate(encoding['offset_mapping']):
                # Check if token overlaps with entity start
                if entity['start'] < end and entity['end'] > start:
                    start_token = i
                # Check if token overlaps with entity end
                if start < entity['end'] <= end and end_token is None:
                    end_token = i
            
            if start_token is not None and end_token is not None:
                # Mark first token as B-ENTITY
                ner_labels[start_token] = 1 if entity['type'] == 'PERSON' else 3
                # Mark subsequent tokens as I-ENTITY
                for i in range(start_token + 1, end_token + 1):
                    ner_labels[i] = 2 if entity['type'] == 'PERSON' else 4
                
                token_entities.append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type'],
                    'id': entity['id']  # Store original ID for relation mapping
                })
        
        # Prepare relation data
        rel_data = {
            'entities': token_entities,
            'pairs': [],
            'labels': []
        }
        
        # Create mapping from entity ID to token entity index
        token_entity_id_to_idx = {e['id']: i for i, e in enumerate(token_entities)}
        
        for relation in sample['relations']:
            # Get token entity indices for relation arguments
            arg1_token_idx = token_entity_id_to_idx.get(relation['arg1'], -1)
            arg2_token_idx = token_entity_id_to_idx.get(relation['arg2'], -1)
            
            # Only add if both entities were successfully tokenized
            if arg1_token_idx != -1 and arg2_token_idx != -1:
                rel_data['pairs'].append((arg1_token_idx, arg2_token_idx))
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

    # В функции обучения добавьте веса классов
    relation_weights = torch.tensor([1.0, 1.0, 0.1])  # Уменьшите вес для NO_RELATION
    rel_loss_fct = nn.CrossEntropyLoss(weight=relation_weights)

    sample = train_dataset[0]
    print("\nSample check:")
    print(f"Input IDs: {sample['input_ids']}")
    print(f"NER labels: {sample['ner_labels']}")
    print(f"Relation pairs: {sample['rel_data']['pairs']}")
    print(f"Relation labels: {sample['rel_data']['labels']}")
    
    for epoch in range(2):
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

def print_entities_details(entities, tokens):
    print("\nPredicted entities (detailed):")
    for e in entities:
        # Получаем токены сущности с правильной обработкой ##
        entity_tokens = []
        for i in e['token_ids']:
            token = tokens[i]
            if token.startswith('##'):
                entity_tokens[-1] += token[2:]  # Объединяем с предыдущим токеном
            else:
                entity_tokens.append(token)
        
        # Восстанавливаем оригинальный текст с пробелами
        original_text = ' '.join(entity_tokens).replace(' ##', '')
        
        print(f"{e['type']}:")
        print(f"  Tokens: {' '.join([tokens[i] for i in e['token_ids']])}")
        print(f"  Token IDs: {e['start']}-{e['end']}")
        print(f"  Reconstructed text: '{original_text}'")
        print(f"  Raw text in dict: '{e['text']}'")
        print("-" * 50)

def predict(text, model, tokenizer, device="cuda"):
    # Токенизация с возвратом смещений
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    # print("\nTokenization details:")
    # for i, (token, (start, end)) in enumerate(zip(tokens, encoding['offset_mapping'][0])):
    #     print(f"{i:3d}: {token:15s} (chars: {start:3d}-{end:3d}) -> '{text[start:end]}'")
    
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
    
    # Декодирование NER
    ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)[0].cpu().numpy()
    
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, ner_preds)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        # Определяем тип сущности и является ли она началом
        if pred == 1:  # B-PER
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': "PERSON",
                'start': i,
                'end': i,
                'token_ids': [i],
                'text': token.replace('##', '')
            }
        elif pred == 2:  # I-PER
            if current_entity and current_entity['type'] == "PERSON":
                if token.startswith('##'):
                    current_entity['text'] += token[2:]
                else:
                    current_entity['text'] += ' ' + token
                current_entity['end'] = i
                current_entity['token_ids'].append(i)
        elif pred == 3:  # B-PROF
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': "PROFESSION",
                'start': i,
                'end': i,
                'token_ids': [i],
                'text': token.replace('##', '')
            }
        elif pred == 4:  # I-PROF
            if current_entity and current_entity['type'] == "PROFESSION":
                if token.startswith('##'):
                    current_entity['text'] += token[2:]
                else:
                    current_entity['text'] += ' ' + token
                current_entity['end'] = i
                current_entity['token_ids'].append(i)
        else:  # O
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)

     # Фильтрация сущностей - удаляем слишком короткие
    entities = [e for e in entities if 
           (e['type'] == 'PERSON' and len(e['token_ids']) >= 1 and 
            not any(t in e['text'].lower() for t in ['.', ',', ';'])) or 
           (e['type'] == 'PROFESSION' and len(e['token_ids']) >= 1 and
            not any(t in e['text'].lower() for t in ['.', ',', ';']))]
    
    # Восстановление оригинального текста сущностей
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    for entity in entities:
        start_char = offset_mapping[entity['start']][0]
        end_char = offset_mapping[entity['end']][1]
        entity['text'] = text[start_char:end_char]

    # Вывод подробной информации о сущностях
    print_entities_details(entities, tokens)
    
    # Извлечение отношений
    relations = []
    
    if len(entities) >= 2:
        sequence_output = model.bert(
            encoding['input_ids'], 
            encoding['attention_mask']
        ).last_hidden_state
        
        # Create entity embeddings
        entity_embeddings = []
        for entity in entities:
            embed = sequence_output[0, entity['start']:entity['end']+1].mean(dim=0)
            entity_embeddings.append(embed)
        
        # Only consider PERSON-PROFESSION pairs that are close in text
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i != j and ((e1['type'] == 'PERSON' and e2['type'] == 'PROFESSION') or 
                              (e1['type'] == 'PERSON' and e2['type'] == 'PERSON')):
                    
                    # Check proximity - max 10 tokens apart
                    distance = abs(e1['start'] - e2['start'])
                    if distance > 10:
                        continue
                        
                    feature = torch.cat([
                        entity_embeddings[i],
                        entity_embeddings[j],
                        sequence_output.mean(dim=1)[0]  # context
                    ], dim=-1)
                    
                    with torch.no_grad():
                        rel_prob = torch.softmax(model.rel_classifier(feature.unsqueeze(0)), dim=-1)[0]
                        pred_label = rel_prob.argmax().item()
                        confidence = rel_prob.max().item()
                    
                    # Only add if high confidence and meaningful relation
                    if confidence > 0.85:  # Higher threshold
                        rel_type = "WORKS_AS" if pred_label == 0 else "WORKPLACE" if pred_label == 1 else None
                        if rel_type:
                            relations.append({
                                'type': rel_type,
                                'arg1': e1,
                                'arg2': e2,
                                'confidence': confidence
                            })
    
    # Вывод информации об отношениях
    print("\nPredicted relations:")
    for r in relations:
        arg1_text = ' '.join([tokens[i] for i in r['arg1']['token_ids']]).replace(' ##', '')
        arg2_text = ' '.join([tokens[i] for i in r['arg2']['token_ids']]).replace(' ##', '')
        print(f"{r['type']}: '{arg1_text}' -> '{arg2_text}' (conf: {r['confidence']:.2f})")
    
    return {
        'text': text,
        'entities': entities,
        'relations': relations
    }

# Usage example
if __name__ == "__main__":
    model, tokenizer = train_model()
    
    test_texts = [
        "Айрат Мурзагалиев, заместитель начальника управления президента РФ, встретился с главой администрации Уфы.",
        "Иван Петров работает программистом в компании Яндекс.",
        "Доктор Сидоров принял пациентку Ковалеву в городской больнице."
    ]
    
    for text in test_texts:
        print("\n" + "="*80)
        print(f"Processing text: '{text}'")
        result = predict(text, model, tokenizer)
        print("\nFinal result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
