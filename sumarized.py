from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class T5Config:
    model_name: str = "t5-small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_input_length: int = 512
    max_output_length: int = 128


class T5Summarizer:
    """
    Minimal T5 wrapper for inference / later integration.
    This file only loads t5-small and provides helper methods.
    Training logic is intentionally omitted so you can write
    your own train pipeline with the 3 files.
    """

    def __init__(self, config: Optional[T5Config] = None):
        self.config = config or T5Config()
        self.device = self.config.device

        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded on: {self.device}")

    def build_input(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize"
    ) -> str:
        """
        Build input text in one of two modes:
        1) Plain seq2seq: input_text
        2) Retrieval-augmented: query + contexts
        """
        if input_text is not None:
            return f"{task_prefix}: {input_text.strip()}"

        parts = []
        if query:
            parts.append(f"question: {query.strip()}")
        if contexts:
            clean_contexts = [c.strip() for c in contexts if c and c.strip()]
            if clean_contexts:
                parts.append("context: " + " ".join(clean_contexts))

        if not parts:
            raise ValueError("Provide either input_text or query/contexts.")

        return f"{task_prefix}: " + " ".join(parts)

    def tokenize(self, text: str):
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
            padding=False,
        )

    def generate(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        max_output_length: Optional[int] = None,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate summary/text from either:
        - input_text
        - query + retrieved contexts
        """
        source_text = self.build_input(
            query=query,
            contexts=contexts,
            input_text=input_text,
            task_prefix=task_prefix,
        )

        inputs = self.tokenize(source_text)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_output_length or self.config.max_output_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load_from_path(cls, model_path: str, device: Optional[str] = None):
        obj = cls.__new__(cls)
        obj.config = T5Config(model_name=model_path)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.tokenizer = AutoTokenizer.from_pretrained(model_path)
        obj.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        obj.model.to(obj.device)
        obj.model.eval()
        return obj


