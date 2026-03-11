from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

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
    T5 wrapper for:
    1) plain summarization
    2) retrieval-augmented summarization
    3) retrieval-aware training

    Compatible with:
    - contexts: List[str]
    - retrieved_items: list of SearchResult-like objects where:
        item.document.text
        item.document.metadata
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

    @staticmethod
    def _safe_strip(text: Optional[str]) -> str:
        if text is None:
            return ""
        return str(text).strip()

    def _format_contexts(
        self,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> List[str]:
        """
        Convert either raw context strings or retrieved_items into
        a cleaned list of context strings.
        """
        formatted_contexts: List[str] = []

        if contexts:
            clean_contexts = [self._safe_strip(c) for c in contexts if self._safe_strip(c)]
            if max_contexts is not None:
                clean_contexts = clean_contexts[:max_contexts]
            return clean_contexts

        if retrieved_items:
            items = retrieved_items[:max_contexts] if max_contexts is not None else retrieved_items

            for item in items:
                doc = getattr(item, "document", None)

                if doc is None:
                    continue

                text = self._safe_strip(getattr(doc, "text", ""))
                metadata = getattr(doc, "metadata", {}) or {}

                if not text:
                    continue

                if include_section_headers:
                    section = self._safe_strip(metadata.get("section", ""))
                    if section:
                        text = f"[{section}] {text}"

                formatted_contexts.append(text)

        return formatted_contexts

    def build_input(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> str:
        """
        Build source text in one of three modes:

        1) Plain seq2seq:
            summarize: <input_text>

        2) Retrieval-augmented with raw contexts:
            summarize: question: ... context: ...

        3) Retrieval-augmented with retrieved_items:
            summarize: question: ... context: [Section] ...
        """
        if input_text is not None:
            clean_input = self._safe_strip(input_text)
            if not clean_input:
                raise ValueError("input_text is empty.")
            return f"{task_prefix}: {clean_input}"

        parts: List[str] = []

        clean_query = self._safe_strip(query)
        if clean_query:
            parts.append(f"question: {clean_query}")

        final_contexts = self._format_contexts(
            contexts=contexts,
            retrieved_items=retrieved_items,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
        )

        if final_contexts:
            parts.append("context: " + " ".join(final_contexts))

        if not parts:
            raise ValueError(
                "Provide either input_text, or query + contexts/retrieved_items."
            )

        return f"{task_prefix}: " + " ".join(parts)

    def tokenize(
        self,
        text: str,
        padding: bool = False,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=True,
            max_length=self.config.max_input_length,
            padding=padding,
        )

    def tokenize_pair(
        self,
        source_text: str,
        target_text: str,
        padding: Union[bool, str] = "max_length",
    ) -> Dict[str, Any]:
        """
        Tokenize source/target pair for training.
        """
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=padding,
        )

        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.config.max_output_length,
            truncation=True,
            padding=padding,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def build_training_example(
        self,
        target_text: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Build a train-ready example:
        {
            "input_text": ...,
            "target_text": ...
        }
        """
        source_text = self.build_input(
            query=query,
            contexts=contexts,
            retrieved_items=retrieved_items,
            input_text=input_text,
            task_prefix=task_prefix,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
        )

        clean_target = self._safe_strip(target_text)
        if not clean_target:
            raise ValueError("target_text is empty.")

        return {
            "input_text": source_text,
            "target_text": clean_target,
        }

    def generate(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
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
        - query + contexts
        - query + retrieved_items
        """
        source_text = self.build_input(
            query=query,
            contexts=contexts,
            retrieved_items=retrieved_items,
            input_text=input_text,
            task_prefix=task_prefix,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
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