# KorQuAD를 불러오는 class입니다.

from typing import Dict, Any
import json
import random

class KoqMRC:
    def __init__(self, data, indices):
        self._data = data
        self._indices = indices

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

        guid = qa['id']    # ['guid'] 질문의 고유번호가 바뀐것을 제외하고는 차이가 없음
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