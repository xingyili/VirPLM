import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM, get_linear_schedule_with_warmup

from utils.data_processing import AminoAcidDataset, CustomDataCollator, load_sequences_from_files
from utils.utils import get_device, set_random_seed



def run_pretraining(cfg, seed=3407, device="auto", seq_len=328):
    set_random_seed(seed)
    device = get_device(device)
    model_name = cfg["base_model_name"]

    config = EsmConfig.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.get("mlm_probability", 0.15),
    )
    sequences = load_sequences_from_files(*cfg["fasta_paths"])
    dataset = AminoAcidDataset(sequences, tokenizer, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: x,
    )

    epochs = cfg.get("epochs", 10)
    accumulation_steps = cfg.get("accumulation_steps", 4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 5e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(cfg.get("warmup_ratio", 0.1) * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    for epoch in range(epochs):
        optimizer.zero_grad()
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", total=len(dataloader))
        for i, batch in progress_bar:
            batch = collator(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (i + 1) % accumulation_steps == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        if (i + 1) % accumulation_steps != 0:
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(dataloader):.4f}")

    save_path = cfg.get("output_dir", "H3pre_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f" Pretrained model saved to {save_path}")
