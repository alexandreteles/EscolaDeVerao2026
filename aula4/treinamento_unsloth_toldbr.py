"""Treinamento com Unsloth para classificação de toxicidade (TOLD-BR).

Exemplo de uso:
python aula4/treinamento_unsloth_toldbr.py \
    --model-name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --output-dir outputs/toldbr-qlora
"""

from __future__ import annotations

import argparse
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """### Instrução:
Classifique a mensagem abaixo como TOXICA ou NAO_TOXICA.

### Mensagem:
{texto}

### Resposta:
{rotulo}"""


TEXT_CANDIDATES = ("text", "texto", "tweet", "sentence", "comment", "content")
LABEL_CANDIDATES = ("label", "toxic", "toxicity", "classe", "class", "target")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina um classificador de toxicidade com Unsloth + TOLD-BR")
    parser.add_argument("--dataset-name", default="JAugusto97/told-br", help="Dataset no Hugging Face Hub")
    parser.add_argument("--dataset-config", default=None, help="Configuração do dataset (se houver)")
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit", help="Modelo base")
    parser.add_argument("--output-dir", default="outputs/toldbr-unsloth", help="Diretório de saída")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1, help="Usado se o dataset não vier com split de validação")
    return parser.parse_args()



def find_column(columns: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    raise ValueError(
        f"Não foi possível identificar a coluna de {kind}. Colunas disponíveis: {list(columns)}"
    )



def normalize_label(value: int | bool | str) -> str:
    if isinstance(value, bool):
        return "TOXICA" if value else "NAO_TOXICA"

    if isinstance(value, int):
        return "TOXICA" if value == 1 else "NAO_TOXICA"

    text_value = str(value).strip().lower()
    toxic_aliases = {"1", "toxic", "toxica", "tóxica", "hate", "abusive"}
    return "TOXICA" if text_value in toxic_aliases else "NAO_TOXICA"



def format_example(example: dict, text_column: str, label_column: str) -> dict[str, str]:
    texto = str(example[text_column]).strip()
    rotulo = normalize_label(example[label_column])
    return {"text": PROMPT_TEMPLATE.format(texto=texto, rotulo=rotulo)}



def ensure_train_eval(dataset: Dataset | DatasetDict, test_size: float, seed: int) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        if "train" in dataset and "validation" in dataset:
            return DatasetDict(train=dataset["train"], validation=dataset["validation"])
        if "train" in dataset and "test" in dataset:
            return DatasetDict(train=dataset["train"], validation=dataset["test"])
        if "train" in dataset:
            split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
            return DatasetDict(train=split["train"], validation=split["test"])

    if isinstance(dataset, Dataset):
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict(train=split["train"], validation=split["test"])

    raise ValueError("Não foi possível construir splits de treino/validação para o dataset informado.")



def main() -> None:
    args = parse_args()

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config)
    dataset = ensure_train_eval(raw_dataset, test_size=args.test_size, seed=args.seed)

    text_column = find_column(dataset["train"].column_names, TEXT_CANDIDATES, "texto")
    label_column = find_column(dataset["train"].column_names, LABEL_CANDIDATES, "rótulo")

    train_dataset = dataset["train"].map(
        format_example,
        fn_kwargs={"text_column": text_column, "label_column": label_column},
        remove_columns=dataset["train"].column_names,
        desc="Formatando treino",
    )
    eval_dataset = dataset["validation"].map(
        format_example,
        fn_kwargs={"text_column": text_column, "label_column": label_column},
        remove_columns=dataset["validation"].column_names,
        desc="Formatando validação",
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        seed=args.seed,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Treinamento concluído. Artefatos em: {args.output_dir}")


if __name__ == "__main__":
    main()
