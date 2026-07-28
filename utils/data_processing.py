import random
import re
from datetime import date

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling


def normalize_strain_name(name):
    text = str(name).strip().lower()
    text = text.split("|")[0].strip()
    return re.sub(r"\s+", " ", text)


def parse_collection_date(value): 
    
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        y, m, d = map(int, text.split("-"))
        return date(y, m, d)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        y, m = map(int, text.split("-"))
        return date(y, m, 15)
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 7, 1)
    raise ValueError(f"Unsupported collection date: {value!r}")


def load_metadata_dates(metadata_path):
    metadata = pd.read_csv(metadata_path)
    required = {"header", "Collection_Date"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata file {metadata_path} missing columns: {sorted(missing)}"
        )

    mapping = {}
    for row in metadata.itertuples(index=False):
        key = normalize_strain_name(getattr(row, "header"))
        try:
            parsed = parse_collection_date(getattr(row, "Collection_Date"))
        except (TypeError, ValueError):
            continue
        mapping.setdefault(key, parsed)
    return mapping


class H3N2DataProcessor:
    replace_dict = {
        "B": "DN",
        "J": "IL",
        "Z": "EQ",
        "X": "ACDEFGHIKLMNPQRSTVWY",
    }

    def __init__(
        self,
        ha_path,
        hi_path,
        bin_threshold=2.0,
        deduplicate=False,
        metadata_path=None,
        extra_ha_path=None,
        extra_hi_path=None,
    ):
        self.ha_path = ha_path
        self.hi_path = hi_path
        self.extra_ha_path = extra_ha_path
        self.extra_hi_path = extra_hi_path
        self.bin_threshold = float(bin_threshold)
        self.deduplicate = deduplicate

        if metadata_path is None:
            raise ValueError(
                "metadata_path is required for retrospective time-split evaluation."
            )
        self.metadata_dates = load_metadata_dates(metadata_path)
        self._preprocess()

    @staticmethod
    def _replace_ambiguous(seq, replace_dict):
        return "".join(
            random.choice(replace_dict[aa]) if aa in replace_dict else aa
            for aa in str(seq)
        )

    def _label_from_distance(self, distance):
        return int(float(distance) >= self.bin_threshold)

    def _resolve_date(self, name, year, source_tag):
        key = normalize_strain_name(name)
        if key in self.metadata_dates:
            return self.metadata_dates[key], "metadata"

        if "smith" in source_tag.lower():
            return date(int(year), 7, 1), "smith_year_midpoint"

        return None, None

    def _process_single_pair(self, ha_path, hi_path, source_tag):
        ha_df = pd.read_csv(ha_path)
        hi_df = pd.read_csv(hi_path)

        required_ha = {"index", "name", "year", "seq"}
        required_hi = {"at_index", "sr_index", "distance"}
        missing_ha = required_ha - set(ha_df.columns)
        missing_hi = required_hi - set(hi_df.columns)
        if missing_ha:
            raise ValueError(f"{ha_path} missing columns: {sorted(missing_ha)}")
        if missing_hi:
            raise ValueError(f"{hi_path} missing columns: {sorted(missing_hi)}")

        ha_records = {}
        for row in ha_df.itertuples(index=False):
            collection_date, date_source = self._resolve_date(
                getattr(row, "name"),
                getattr(row, "year"),
                source_tag,
            )
            if collection_date is None:
                continue
            ha_records[int(getattr(row, "index"))] = {
                "name": getattr(row, "name"),
                "seq": getattr(row, "seq"),
                "year": int(getattr(row, "year")),
                "date": collection_date,
                "date_source": date_source,
            }

        entries = []
        for row in hi_df.itertuples(index=False):
            ai = int(getattr(row, "at_index"))
            si = int(getattr(row, "sr_index"))
            if ai not in ha_records or si not in ha_records:
                continue

            at = ha_records[ai]
            sr = ha_records[si]
            distance = float(getattr(row, "distance"))
            entries.append(
                {
                    "at_name": at["name"],
                    "sr_name": sr["name"],
                    "at_seq": at["seq"],
                    "sr_seq": sr["seq"],
                    "at_year": at["year"],
                    "sr_year": sr["year"],
                    "at_date": at["date"],
                    "sr_date": sr["date"],
                    "at_date_source": at["date_source"],
                    "sr_date_source": sr["date_source"],
                    "distance": distance,
                    "class_label": self._label_from_distance(distance),
                    "source": source_tag,
                }
            )
        return entries

    def _preprocess(self):
        print("Processing data...")
        entries = self._process_single_pair(
            self.ha_path, self.hi_path, "H3N2"
        )

        if self.extra_ha_path and self.extra_hi_path:
            entries.extend(
                self._process_single_pair(
                    self.extra_ha_path,
                    self.extra_hi_path,
                    "H3N2_smith",
                )
            )

        if self.deduplicate:
            seen = set()
            deduped = []
            for entry in entries:
                key = (entry["at_name"], entry["sr_name"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(entry)
            entries = deduped

        self.at_name = [e["at_name"] for e in entries]
        self.sr_name = [e["sr_name"] for e in entries]
        self.at_seq = [
            self._replace_ambiguous(e["at_seq"], self.replace_dict)
            for e in entries
        ]
        self.sr_seq = [
            self._replace_ambiguous(e["sr_seq"], self.replace_dict)
            for e in entries
        ]
        self.distances = [float(e["distance"]) for e in entries]
        self.classes = [int(e["class_label"]) for e in entries]
        self.sources = [e["source"] for e in entries]
        self.at_years = [e["at_year"] for e in entries]
        self.sr_years = [e["sr_year"] for e in entries]
        self.at_dates = [e["at_date"] for e in entries]
        self.sr_dates = [e["sr_date"] for e in entries]
        self.at_date_sources = [e["at_date_source"] for e in entries]
        self.sr_date_sources = [e["sr_date_source"] for e in entries]


class FLUDataset(Dataset):
    def __init__(self, processor, tokenizer, seq_len=328, mode="train"):
        self.at_name = processor.at_name
        self.sr_name = processor.sr_name
        self.at_seq = processor.at_seq
        self.sr_seq = processor.sr_seq
        self.at_years = processor.at_years
        self.sr_years = processor.sr_years
        self.at_dates = processor.at_dates
        self.sr_dates = processor.sr_dates
        self.at_date_sources = processor.at_date_sources
        self.sr_date_sources = processor.sr_date_sources
        self.distances = processor.distances
        self.classes = processor.classes
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        seq1, seq2 = self.at_seq[idx], self.sr_seq[idx]
        if self.mode == "train":
            seq1 = self.augment(seq1)
            seq2 = self.augment(seq2)
        input_ids, attention_mask = encode_pair(
            self.tokenizer, seq1, seq2, self.seq_len * 2
        )
        y_dist = torch.tensor(self.distances[idx], dtype=torch.float32)
        y_class = torch.tensor(self.classes[idx], dtype=torch.long)
        return input_ids.squeeze(0), attention_mask.squeeze(0), y_dist, y_class

    @staticmethod
    def augment(
        seq,
        delete_prob=0.02,
        substitute_prob=0.03,
        insert_prob=0.01,
    ):
        aas = "ACDEFGHIKLMNPQRSTVWY"
        seq = [aa for aa in seq if random.random() > delete_prob]
        seq = [
            random.choice(aas) if random.random() < substitute_prob else aa
            for aa in seq
        ]
        out = []
        for aa in seq:
            out.append(aa)
            if random.random() < insert_prob:
                out.append(random.choice(aas))
        return "".join(out)


def encode_pair(tokenizer, seq1, seq2, max_length):
    enc = tokenizer(
        f"{seq1}[SEP]{seq2}",
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    return enc["input_ids"], enc["attention_mask"]


class AminoAcidDataset(Dataset):
    def __init__(self, sequences, tokenizer, seq_len):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            add_special_tokens=True,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding


def load_sequences_from_files(*file_names):
    sequences = []
    for file_path in file_names:
        with open(file_path, "r", encoding="utf-8") as file:
            current = []
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current:
                        sequences.append("".join(current))
                        current = []
                else:
                    current.append(line)
            if current:
                sequences.append("".join(current))
    return sequences


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        special_tokens_mask = batch["attention_mask"] == 0
        batch["input_ids"] = batch["input_ids"].masked_fill(
            special_tokens_mask, value=0
        )
        return batch
