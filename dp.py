def is_running_on_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

if is_running_on_ipython():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch

import re
from typing import List, Tuple, Dict, Any
import json
import random

class KoMRC:
    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices

    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as fd:
            data = json.load(fd)

        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))
        
        return cls(data, indices)

    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, eval_ratio: float=.1, seed=42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * eval_ratio):]
        eval_indices = indices[:int(len(indices) * eval_ratio)]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]

        context = paragraph['context']
        qa = paragraph['qas'][q_id]

        guid = qa['guid']
        question = qa['question']
        answers = qa['answers']

        return {
            'guid': guid,
            'context': context,
            'question': question,
            'answers': answers
        }

    def __len__(self) -> int:
        return len(self._indices)

class Preprocessor:
    def __init__(self, tokenizer, max_len, doc_stride, padding_side):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_stride = doc_stride
        self.padding_side = padding_side
        self.dataset = None

    def load_dataset(self, dataset, eval=False):
        _guid = list(map(lambda x: x['guid'], dataset))
        _context = list(map(lambda x: x['context'], dataset))
        _question = list(map(lambda x: x['question'], dataset))
        if not eval:
            _answers = list(map(lambda x: x['answers'], dataset))

        if eval:
            assert len(_guid) == len(_context) == len(_question)
            self.dataset = {'guid': _guid, 
                            'context': _context, 
                            'question': _question
                            }
        else:
            assert len(_guid) == len(_context) == len(_question) == len(_answers)
            self.dataset = {'guid': _guid, 
                            'context': _context, 
                            'question': _question,
                            'answers': _answers
                            }
            
    def tokenize(self):
        output = []
        print(f'Tokenizing...')
        pad_on_right = self.tokenizer.padding_side == "right"

        # Tokenization
        tokenized_examples = self.tokenizer(
                self.dataset["question" if pad_on_right else "context"],
                self.dataset["context" if pad_on_right else "question"],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=self.max_len,
                stride=self.doc_stride, 
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
        
        # guid matching
        _guid = list(map(lambda x: self.dataset['guid'][x], tokenized_examples['overflow_to_sample_mapping']))
        tokenized_examples['guid'] = _guid

        sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        offset_mapping = tokenized_examples["offset_mapping"]

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in tqdm(enumerate(offset_mapping), total=len(tokenized_examples['input_ids'])):
            # 답이 존재하지 않는 경우 answer에 대한 start, end가 [CLS]를 가르키도록 설정할 것이다.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # sequence_ids를 통해 question와 context를 구분할 수 있다.
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            ### 정답이 여러 개인 경우 어떻게 해결해야할까????????????
            answers = self.dataset["answers"][sample_index][0]

            # 정답이 없는 질문일 경우
            if len(answers['text']) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 정답이 지문 내에 존재한다면
                start_char = answers["answer_start"]
                end_char = start_char + len(answers["text"])

                # start, end를 찾는 이전 코드와 같다.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 해당 지문에 존재하는지 확인한다.
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        for i in range(len(tokenized_examples['overflow_to_sample_mapping'])):
            example = {}
            for key in tokenized_examples.keys():
                example[key] = tokenized_examples[key][i]
            output.append(example)
            
        return output

    def tokenize_eval(self):
        output = []
        print(f'Tokenizing...')
        pad_on_right = self.tokenizer.padding_side == "right"

        # Tokenization
        tokenized_examples = self.tokenizer(
                self.dataset["question" if pad_on_right else "context"],
                self.dataset["context" if pad_on_right else "question"],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=self.max_len,
                stride=self.doc_stride, 
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
        
        # guid matching
        _guid = list(map(lambda x: self.dataset['guid'][x], tokenized_examples['overflow_to_sample_mapping']))
        tokenized_examples['guid'] = _guid

        return tokenized_examples

def collator(_samples):
    keys = _samples[0].keys()
    samples = dict()
    for key in keys:
        samples[key] = []

    for _sample in _samples:
        for key in keys:
            samples[key].append(_sample[key])

    for key in ['input_ids', 'attention_mask', 'token_type_ids', 'offset_mapping', 'start_positions', 'end_positions']:
        samples[key] = torch.tensor(samples[key], dtype=torch.long)

    return samples

# Metric
def compute_levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]

class Postprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def decode_pred_and_true(self, pred_ids, true_ids):
        pred_answer = list(map(lambda x: self.tokenizer.decode(x), pred_ids))
        true_answer = list(map(lambda x: self.tokenizer.decode(x), true_ids))
        return pred_answer, true_answer

    def postprocess(self, input_ids, pred, true, dist=True, max_len=20):
        pred_s, pred_e = pred
        true_s, true_e = true

        batch_size = len(pred_s)
        # end가 start보다 먼저 나오면, 답을 찾을 수 없는 경우로 판단합니다.
        
        for i in range(batch_size):
            if pred_s[i] > pred_e[i]:
                pred_s[i] = 0
                pred_e[i] = 0     

        pred_ids = []
        true_ids = []
        # start, end로 answer string을 구합니다.
        for i in range(batch_size):
            pred_ids.append(input_ids[i][pred_s[i]:pred_e[i]+1])
            true_ids.append(input_ids[i][true_s[i]:true_e[i]+1])
        pred_answer, true_answer = self.decode_pred_and_true(pred_ids, true_ids)
        
        # pred answer의 길이가 max_len을 초과하는지 확인합니다.
        if max_len:
            pred_answer = list(map(lambda x: x if len(x) <= max_len else '', pred_answer))       

        # answer에서 special token을 삭제합니다.
        pred_answer = list(map(self.remove_tokens, pred_answer))
        true_answer = list(map(self.remove_tokens, true_answer))

        if dist:
            levenshtein_distances = list(map(lambda x: self.levenshtein_distance(x[0], x[1]), zip(pred_answer, true_answer)))
            return pred_answer, true_answer, levenshtein_distances
        else:
            return pred_answer, true_answer

    def postprocess_eval(self, input_ids, pred):
        pred_s, pred_e = pred

        batch_size = len(pred_s)
        # end가 start보다 먼저 나오면, 답을 찾을 수 없는 경우로 판단합니다.
        
        for i in range(batch_size):
            if pred_s[i] > pred_e[i]:
                pred_s[i] = 0
                pred_e[i] = 0     

        pred_ids = []
        # start, end로 answer string을 구합니다.
        for i in range(batch_size):
            pred_ids.append(input_ids[i][pred_s[i]:pred_e[i]+1])
        pred_answer = pred_answer = list(map(lambda x: self.tokenizer.decode(x), pred_ids))

        return pred_answer
    
    def remove_tokens(self, answer):
        p = re.compile(r'(\[CLS\]|\[SEP\]|\[PAD\]|#)')
        answer = p.sub('', answer)
        answer.strip()
        return answer

if __name__ == '__main__':
    s1 = '안녕하세요'
    s2 = '오 안녕하세'
    print(compute_levenshtein(s1, s2, True))