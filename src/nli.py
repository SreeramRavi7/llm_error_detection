"""Optional NLI clustering for TriviaQA (the original semantic entropy method, Farquhar et al. 2024).

Two answers are in the same cluster if each entails the other, judged with the question
as context. Uses microsoft/deberta-large-mnli. GSM8K never needs this, numbers are compared directly.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .grading import normalize


class NLIClusterer:
    # float32 on purpose: DeBERTa v1 attention breaks in fp16 ("expected Float but found Half")
    def __init__(self, name="microsoft/deberta-large-mnli"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    @torch.no_grad()
    def _entails(self, pairs):
        enc = self.tok([p for p, _ in pairs], [h for _, h in pairs], return_tensors="pt",
                       padding=True, truncation=True, max_length=256).to(self.model.device)
        return (self.model(**enc).logits.argmax(-1) == self.entail).tolist()

    def cluster(self, question, answers):
        ids, reps = [], []
        for ans in answers:
            match = next((i for i, r in enumerate(reps) if normalize(r) == normalize(ans)), None)
            if match is None and reps:
                a = f"{question} {ans}"
                pairs = []
                for r in reps:
                    b = f"{question} {r}"
                    pairs += [(a, b), (b, a)]
                ent = self._entails(pairs)
                match = next((i for i in range(len(reps)) if ent[2 * i] and ent[2 * i + 1]), None)
            if match is None:
                reps.append(ans)
                match = len(reps) - 1
            ids.append(match)
        return ids
