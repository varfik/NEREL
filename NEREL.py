# 1. Клонируем репозиторий (если еще не сделали)
# !git clone https://github.com/nerel-ds/NEREL.git

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.nn.utils.rnn import pad_sequence

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=3, num_rel_labels=3):
        super().__init__()
        self.num_ner_labels = num_ner_labels
        self.num_rel_labels = num_rel_labels

        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        # NER Head
        self.ner_classifier = nn.Linear(self.config.hidden_size, num_ner_labels)

        # Relation Head
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_rel_labels)
        )

    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        ner_logits = self.ner_classifier(sequence_output)
        total_loss = 0
        rel_logits = None

        # NER loss calculation
        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_ner_labels)[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = loss_fct(active_logits, active_labels)
            total_loss += ner_loss

        # Relation processing
        if rel_data is not None and isinstance(rel_data, list):
            # Собираем все сущности и отношения по батчу
            all_entities = []
            all_pairs = []
            all_labels = []

            for sample in rel_data:
                if not isinstance(sample, dict):
                    continue

                entities = sample.get('entities', [])
                pairs = sample.get('pairs', [])
                labels = sample.get('labels', [])

                offset = len(all_entities)
                all_entities.extend(entities)
                all_pairs.extend([(p[0]+offset, p[1]+offset) for p in pairs])
                all_labels.extend(labels)

            # Если есть хотя бы одна пара
            if all_pairs:
                # Получаем эмбеддинги для всех сущностей
                entity_embeddings = []
                for entity in all_entities:
                    start = min(entity['start'], sequence_output.size(1)-1)
                    end = min(entity['end'], sequence_output.size(1)-1)
                    # [batch_size, hidden_size]
                    entity_embed = sequence_output[:, start:end+1].mean(dim=1)
                    entity_embeddings.append(entity_embed)

                # Подготавливаем фичи для отношений
                rel_features = []
                batch_size = sequence_output.size(0)
                hidden_size = sequence_output.size(2)

                for pair in all_pairs:
                    e1 = entity_embeddings[pair[0]]  # [batch_size, hidden_size]
                    e2 = entity_embeddings[pair[1]]  # [batch_size, hidden_size]

                    # Контекст - среднее по всем токенам [batch_size, hidden_size]
                    context = sequence_output.mean(dim=1)

                    # Комбинируем фичи для каждого элемента батча
                    for i in range(batch_size):
                        feature = torch.cat([
                            e1[i],  # [hidden_size]
                            e2[i],  # [hidden_size]
                            context[i]  # [hidden_size]
                        ], dim=-1)  # [hidden_size*3]
                        rel_features.append(feature)

                if rel_features:
                    rel_features = torch.stack(rel_features)  # [batch_size*num_pairs, hidden_size*3]
                    rel_logits = self.rel_classifier(rel_features)

                    # Подготавливаем метки (повторяем для каждого элемента батча)
                    rel_labels = torch.tensor(all_labels,
                                          dtype=torch.long,
                                          device=input_ids.device)
                    rel_labels = rel_labels.repeat(batch_size)

                    # Вычисляем loss
                    rel_loss_fct = nn.CrossEntropyLoss()
                    rel_loss = rel_loss_fct(rel_logits, rel_labels)
                    total_loss += rel_loss

        return {
            'ner_logits': ner_logits,
            'rel_logits': rel_logits,
            'loss': total_loss if total_loss != 0 else torch.tensor(0.0, device=input_ids.device)
        }

from collections import defaultdict
import os
import json

class NERELDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Собираем все .txt файлы в папке
        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]

        for txt_file in txt_files:
            txt_path = os.path.join(self.data_dir, txt_file)
            ann_path = os.path.join(self.data_dir, txt_file.replace('.txt', '.ann'))

            # Проверяем, есть ли соответствующий .ann файл
            if not os.path.exists(ann_path):
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            entities, relations = self._parse_ann_file(ann_path)
            samples.append({
                'text': text,
                'entities': entities,
                'relations': relations
            })
        return samples

    def _parse_ann_file(self, ann_path):
        entities = []
        relations = []

        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                entity_id = parts[0]
                type_and_span = parts[1].split()
                entity_type = type_and_span[0]
                start, end = int(type_and_span[1]), int(type_and_span[-1])
                text = parts[2]

                if entity_type in ['PERSON', 'PROFESSION']:
                    entities.append({
                        'id': entity_id,
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'text': text
                    })

            elif line.startswith('R'):
                parts = line.strip().split('\t')
                rel_type = parts[1].split()[0]
                arg1 = parts[1].split()[1].split(':')[1]
                arg2 = parts[1].split()[2].split(':')[1]

                if rel_type in ['WORKS_AS', 'WORKPLACE']:
                    relations.append({
                        'type': rel_type,
                        'arg1': arg1,
                        'arg2': arg2
                    })

        return entities, relations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        print("\n=== Обработка примера ===")
        print(f"Исходный текст: {sample['text']}")

        # Вывод информации о сущностях
        print("\nСущности в тексте:")
        for entity in sample['entities']:
            print(f"{entity['type']}: {entity['text']} (позиции: {entity['start']}-{entity['end']})")
        text = sample['text']
        entities = sample['entities']
        relations = sample['relations']

        # Tokenize text
        encoding = self.tokenizer(text,
                                max_length=self.max_length,
                                truncation=True,
                                return_offsets_mapping=True)

        # Align entities with tokens
        token_entities = []
        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']

            # Find tokens that overlap with entity span
            start_token = None
            end_token = None
            for i, (start, end) in enumerate(encoding['offset_mapping']):
                if start <= start_char < end:
                    start_token = i
                if start < end_char <= end:
                    end_token = i
                    break

            if start_token is not None and end_token is not None:
                token_entities.append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type']
                })

        # Prepare NER labels
        ner_labels = [0] * len(encoding['input_ids'])  # 0 = O (outside)
        for entity in token_entities:
            ner_labels[entity['start']] = 1 if entity['type'] == 'PERSON' else 2
            for i in range(entity['start'] + 1, entity['end'] + 1):
                ner_labels[i] = ner_labels[entity['start']]

        # Prepare relation data (даже если нет отношений)
        rel_data = {
            'entities': token_entities,
            'pairs': [],
            'labels': []
        }

        # Если есть сущности и отношения
        if len(token_entities) >= 2 and len(relations) > 0:
            entity_map = {e['id']: i for i, e in enumerate(entities)}
            for relation in relations:
                arg1_idx = entity_map.get(relation['arg1'], -1)
                arg2_idx = entity_map.get(relation['arg2'], -1)

                if arg1_idx != -1 and arg2_idx != -1 and arg1_idx < len(token_entities) and arg2_idx < len(token_entities):
                    rel_data['pairs'].append((arg1_idx, arg2_idx))
                    rel_data['labels'].append(0 if relation['type'] == 'WORKS_AS' else 1)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'ner_labels': torch.tensor(ner_labels),
            'rel_data': rel_data  # Всегда содержит 'entities', 'pairs' и 'labels'
        }

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # Добавляем импорт прогресс-бара

from torch.optim import AdamW  # Исправленный импорт
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

def train_model():
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = NERRelationModel(num_ner_labels=3, num_rel_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    def collate_fn(batch):
        # Получаем максимальную длину в батче
        max_len = max(len(item['input_ids']) for item in batch)

        # Применяем паддинг ко всем элементам батча
        input_ids = torch.stack([
            torch.nn.functional.pad(
                item['input_ids'],
                (0, max_len - len(item['input_ids'])),
                value=tokenizer.pad_token_id
            ) for item in batch
        ])

        attention_mask = torch.stack([
            torch.nn.functional.pad(
                item['attention_mask'],
                (0, max_len - len(item['attention_mask'])),
                value=0
            ) for item in batch
        ])

        ner_labels = torch.stack([
            torch.nn.functional.pad(
                item['ner_labels'],
                (0, max_len - len(item['ner_labels'])),
                value=0  # Паддинг для меток NER (0 обычно соответствует 'O')
            ) for item in batch
        ])

        # Обработка rel_data - сохраняем как список словарей
        rel_data = [item['rel_data'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': ner_labels,
            'rel_data': rel_data  # Это будет список словарей
        }
    train_dataset = NERELDataset("/NEREL/NEREL-v1.1/train", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):
        model.train()
        epoch_loss = 0
        ner_correct = 0
        ner_total = 0
        rel_correct = 0
        rel_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'ner_labels': batch['ner_labels'].to(device),
                'rel_data': batch['rel_data']
            }

            outputs = model(**inputs)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Расчет точности NER
            if outputs['ner_logits'] is not None:
                ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)
                active_mask = inputs['attention_mask'] == 1
                ner_correct += ((ner_preds == inputs['ner_labels']) & active_mask).sum().item()
                ner_total += active_mask.sum().item()

            # Расчет точности Relation (исправленная версия)
            if outputs['rel_logits'] is not None:
                rel_preds = torch.argmax(outputs['rel_logits'], dim=-1)

                # Правильно собираем метки отношений
                rel_labels = []
                for sample in batch['rel_data']:
                    if isinstance(sample, dict) and 'labels' in sample:
                        rel_labels.extend(sample['labels'])

                if len(rel_labels) > 0:
                    rel_labels = torch.tensor(rel_labels, device=device)
                    # Важно: учитываем batch_size при сравнении
                    if len(rel_preds) == len(rel_labels):
                        rel_correct += (rel_preds == rel_labels).sum().item()
                        rel_total += len(rel_labels)
                    else:
                        # Логируем несоответствие размеров
                        print(f"Warning: preds {len(rel_preds)} != labels {len(rel_labels)}")

            progress_bar.set_postfix({
                'loss': epoch_loss / (progress_bar.n + 1),
                'NER_acc': f"{ner_correct/ner_total:.2%}" if ner_total > 0 else "N/A",
                'REL_acc': f"{rel_correct/rel_total:.2%}" if rel_total > 0 else "N/A"
            })

        # Статистика по эпохе
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"NER Accuracy: {ner_correct}/{ner_total} ({ner_correct/ner_total:.2%})")
        print(f"Relation Accuracy: {rel_correct}/{rel_total} ({rel_correct/rel_total:.2%})")

    return model, tokenizer


def extract_relations(text, model, tokenizer, device="cpu"):
    # Tokenize input
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Decode NER predictions
    ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)[0].cpu().numpy()

    # Extract entities
    entities = []
    current_entity = None

    for i, (token_id, pred) in enumerate(zip(input_ids[0], ner_preds)):
        if pred != 0:  # Not O
            token = tokenizer.decode([token_id])
            if pred == 1:  # PERSON
                entity_type = "PERSON"
            else:  # PROFESSION
                entity_type = "PROFESSION"

            if current_entity is None or current_entity['type'] != entity_type:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'text': token
                }
            else:
                current_entity['end'] = i
                current_entity['text'] += token

    if current_entity is not None:
        entities.append(current_entity)

    # Extract relations if there are at least 2 entities
    relations = []
    if len(entities) >= 2 and outputs['rel_logits'] is not None:
        # Get all possible entity pairs
        pairs = [(i, j) for i in range(len(entities)) for j in range(len(entities)) if i != j]

        # Prepare relation features
        sequence_output = model.bert(input_ids, attention_mask).last_hidden_state
        context_embed = sequence_output.mean(dim=1)

        rel_features = []
        for pair in pairs:
            e1_embed = sequence_output[:, entities[pair[0]]['start']:entities[pair[0]]['end']].mean(dim=1)
            e2_embed = sequence_output[:, entities[pair[1]]['start']:entities[pair[1]]['end']].mean(dim=1)
            combined = torch.cat([e1_embed, e2_embed, context_embed], dim=-1)
            rel_features.append(combined)

        rel_features = torch.stack(rel_features)
        rel_logits = model.rel_classifier(rel_features)
        rel_preds = torch.argmax(rel_logits, dim=-1).cpu().numpy()

        # Filter only meaningful relations
        for i, pred in enumerate(rel_preds):
            if pred != 2:  # 2 = NO_RELATION (assuming 3 classes)
                e1_idx, e2_idx = pairs[i]
                rel_type = "WORKS_AS" if pred == 0 else "WORKPLACE"
                relations.append({
                    'type': rel_type,
                    'arg1': entities[e1_idx],
                    'arg2': entities[e2_idx]
                })

    return {
        'text': text,
        'entities': entities,
        'relations': relations
    }

# Обучение модели
model, tokenizer = train_model()

# Пример использования
text = "Айрат Мурзагалиев, заместителя начальника управления президента РФ по внутренней политике, встретился с главой администрации Уфы."
result = extract_relations(text, model, tokenizer)

print("Извлеченные сущности:")
for entity in result['entities']:
    print(f"{entity['type']}: {entity['text']}")

print("\nИзвлеченные отношения:")
for rel in result['relations']:
    print(f"{rel['type']}: {rel['arg1']['text']} -> {rel['arg2']['text']}")
