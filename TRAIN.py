import json
import logging
import os
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)


# Data loading


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no} → {e}"
                ) from e
    return records


def ensure_required_fields(records: List[Dict[str, Any]], path: str) -> None:
    required = {"input_text", "target_text"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(
                f"Missing fields {missing} in record {i} of {path}. "
                "Every record must have 'input_text' and 'target_text'."
            )


def build_dataset_dict(
    train_path:      str,
    valid_path:      str,
    test_clean_path: Optional[str] = None,
    test_noisy_path: Optional[str] = None,
) -> DatasetDict:
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)
    ensure_required_fields(train_records, train_path)
    ensure_required_fields(valid_records, valid_path)

    ds_dict: Dict[str, Dataset] = {
        "train":      Dataset.from_list(train_records),
        "validation": Dataset.from_list(valid_records),
    }

    if test_clean_path and os.path.exists(test_clean_path):
        test_clean = load_jsonl(test_clean_path)
        ensure_required_fields(test_clean, test_clean_path)
        ds_dict["test_clean"] = Dataset.from_list(test_clean)

    if test_noisy_path and os.path.exists(test_noisy_path):
        test_noisy = load_jsonl(test_noisy_path)
        ensure_required_fields(test_noisy, test_noisy_path)
        ds_dict["test_noisy"] = Dataset.from_list(test_noisy)

    return DatasetDict(ds_dict)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5 for retrieval-augmented summarisation."
    )

    # Data
    parser.add_argument("--data_dir",        type=str, default="./prepared_data")
    parser.add_argument("--train_file",      type=str, default="train.jsonl")
    parser.add_argument("--valid_file",      type=str, default="valid.jsonl")
    parser.add_argument("--test_clean_file", type=str, default="test_clean.jsonl")
    parser.add_argument("--test_noisy_file", type=str, default="test_noisy.jsonl")

    # Model
    parser.add_argument("--model_name",         type=str, default="t5-small")
    parser.add_argument("--output_dir",         type=str, default="./outputs_t5")
    parser.add_argument("--max_input_length",   type=int, default=512)
    parser.add_argument("--max_target_length",  type=int, default=128)

    # Optimisation
    parser.add_argument("--learning_rate",                 type=float, default=3e-4)
    parser.add_argument("--weight_decay",                  type=float, default=0.01)
    parser.add_argument("--num_train_epochs",              type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size",   type=int,   default=4)
    parser.add_argument("--per_device_eval_batch_size",    type=int,   default=4)
    parser.add_argument("--gradient_accumulation_steps",   type=int,   default=1)
    parser.add_argument("--warmup_ratio",                  type=float, default=0.05)

    # Checkpointing / evaluation
    parser.add_argument("--eval_strategy",        type=str,  default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy",        type=str,  default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--logging_steps",        type=int,  default=50)
    parser.add_argument("--save_total_limit",     type=int,  default=2)
    parser.add_argument("--metric_for_best_model", type=str, default="rougeL")
    parser.add_argument("--greater_is_better",    action="store_true", default=True)

    # Generation
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--generation_num_beams",  type=int, default=4)

    # Hardware / misc
    parser.add_argument("--use_fp16",  action="store_true")
    parser.add_argument("--use_bf16",  action="store_true")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="none")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    train_path      = os.path.join(args.data_dir, args.train_file)
    valid_path      = os.path.join(args.data_dir, args.valid_file)
    test_clean_path = os.path.join(args.data_dir, args.test_clean_file)
    test_noisy_path = os.path.join(args.data_dir, args.test_noisy_file)

    dataset = build_dataset_dict(
        train_path=train_path,
        valid_path=valid_path,
        test_clean_path=test_clean_path,
        test_noisy_path=test_noisy_path,
    )
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=args.max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenising dataset",
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions can be a tuple (e.g. (token_ids, scores)) with some
        # Trainer configurations — take only the token id tensor.
        if isinstance(predictions, tuple):
            logger.warning(
                "compute_metrics received a tuple for `predictions`; "
                "using predictions[0] (shape: %s). "
                "If this is unexpected, check predict_with_generate=True.",
                getattr(predictions[0], "shape", "unknown"),
            )
            predictions = predictions[0]

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v, 4) for k, v in result.items()}

        pred_lens      = [np.count_nonzero(p != tokenizer.pad_token_id) for p in predictions]
        result["gen_len"] = round(float(np.mean(pred_lens)), 4)
        return result

    # ------------------------------------------------------------------
    # Trainer setup
    # ------------------------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if (args.use_fp16 or args.use_bf16) else None,
    )

    # load_best_model_at_end requires both eval and save to be enabled;
    # otherwise the Trainer cannot reload a checkpoint at the end.
    load_best = args.eval_strategy != "no" and args.save_strategy != "no"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        fp16=args.use_fp16,
        bf16=args.use_bf16,
        report_to=args.report_to,
        seed=args.seed,
    )

    callbacks = []
    if args.eval_strategy != "no":
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print("\nStarting training...")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    all_metrics: Dict[str, Any] = {}

    with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, ensure_ascii=False, indent=2)
    all_metrics["train"] = train_result.metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    print("\nValidation evaluation...")
    valid_metrics = trainer.evaluate(
        eval_dataset=tokenized["validation"],
        metric_key_prefix="valid",
    )
    _save_metrics(args.output_dir, "valid_metrics.json", valid_metrics)
    all_metrics["valid"] = valid_metrics

    if "test_clean" in tokenized:
        print("\nTest-clean evaluation...")
        tc_metrics = trainer.evaluate(
            eval_dataset=tokenized["test_clean"],
            metric_key_prefix="test_clean",
        )
        _save_metrics(args.output_dir, "test_clean_metrics.json", tc_metrics)
        all_metrics["test_clean"] = tc_metrics

    if "test_noisy" in tokenized:
        print("\nTest-noisy evaluation...")
        tn_metrics = trainer.evaluate(
            eval_dataset=tokenized["test_noisy"],
            metric_key_prefix="test_noisy",
        )
        _save_metrics(args.output_dir, "test_noisy_metrics.json", tn_metrics)
        all_metrics["test_noisy"] = tn_metrics

    # Combined summary — useful for quick comparisons
    _save_metrics(args.output_dir, "all_metrics.json", all_metrics)

    print("\nDone.")
    print(f"Model saved to: {args.output_dir}")
    _print_metrics_summary(all_metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_metrics(output_dir: str, filename: str, metrics: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _print_metrics_summary(all_metrics: Dict[str, Any]) -> None:
    print("\n── Metrics summary ──────────────────────────────────")
    for split, m in all_metrics.items():
        if not isinstance(m, dict):
            continue
        rouge_l = m.get(f"{split}_rougeL", m.get("rougeL", "N/A"))
        print(f"  {split:<15} rougeL = {rouge_l}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()