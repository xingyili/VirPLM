import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling


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
    ):
        self.ha_df = pd.read_csv(ha_path)
        self.hi_df = pd.read_csv(hi_path)
        self.bin_threshold = float(bin_threshold)
        self.deduplicate = deduplicate
        self._preprocess()

    @staticmethod
    def _replace_ambiguous(seq, replace_dict):
        return "".join(
            [random.choice(replace_dict[aa]) if aa in replace_dict else aa for aa in seq]
        )

    def _label_from_distance(self, d):
        return 0 if d < self.bin_threshold else 1

    def _process_single_pair(self, ha_df, hi_df, source_tag):
        local_entries = []
        if "year" not in ha_df.columns:
            raise ValueError(f"[{source_tag}] HA文件中缺少 'year' 列，无法获取年份信息。")

        ha_df["year"] = pd.to_numeric(ha_df["year"], errors="coerce").fillna(0).astype(int)
        idx2seq = dict(zip(ha_df["index"], ha_df["seq"]))
        idx2name = dict(zip(ha_df["index"], ha_df["name"]))
        idx2year = dict(zip(ha_df["index"], ha_df["year"]))

        mismatch_cnt = 0
        for _, r in hi_df.iterrows():
            ai, si = r["at_index"], r["sr_index"]
            if ai in idx2seq and si in idx2seq:
                at_seq, sr_seq = idx2seq[ai], idx2seq[si]
                at_name, sr_name = idx2name[ai], idx2name[si]
                at_year = idx2year.get(ai, 0)
                sr_year = idx2year.get(si, 0)
                distance = float(r["distance"])
                cls_std = self._label_from_distance(distance)

                try:
                    if "class" in r:
                        cls_orig = int(r["class"])
                        if cls_orig != cls_std:
                            mismatch_cnt += 1
                except Exception:
                    pass

                local_entries.append(
                    {
                        "at_name": at_name,
                        "sr_name": sr_name,
                        "at_seq": at_seq,
                        "sr_seq": sr_seq,
                        "at_year": at_year,
                        "sr_year": sr_year,
                        "distance": distance,
                        "class_label": cls_std,
                        "source": source_tag,
                    }
                )

        if mismatch_cnt > 0:
            print(f"[{source_tag}] 注意: 有 {mismatch_cnt} 条记录原始 class 与统一标准(thr={self.bin_threshold})不一致。")
        return local_entries

    def _preprocess(self):
        entries = []
        print("Processing data...")
        entries.extend(self._process_single_pair(self.ha_df, self.hi_df, "68-24"))

        if self.deduplicate:
            seen = set()
            deduped = []
            for e in entries:
                key = (e["at_name"], e["sr_name"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(e)
            entries = deduped

        self.at_name = [e["at_name"] for e in entries]
        self.sr_name = [e["sr_name"] for e in entries]
        self.at_seq = [self._replace_ambiguous(e["at_seq"], self.replace_dict) for e in entries]
        self.sr_seq = [self._replace_ambiguous(e["sr_seq"], self.replace_dict) for e in entries]
        self.distances = [float(e["distance"]) for e in entries]
        self.classes = [int(e["class_label"]) for e in entries]
        self.sources = [e["source"] for e in entries]
        self.at_years = [e["at_year"] for e in entries]
        self.sr_years = [e["sr_year"] for e in entries]



class FLUDataset(Dataset):
    def __init__(self, processor, tokenizer, seq_len=328, mode="train"):
        self.at_name = processor.at_name
        self.sr_name = processor.sr_name
        self.at_seq = processor.at_seq
        self.sr_seq = processor.sr_seq
        self.at_years = processor.at_years
        self.sr_years = processor.sr_years
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
        input_ids, attention_mask = encode_pair(self.tokenizer, seq1, seq2, self.seq_len * 2)
        y_dist = torch.tensor(self.distances[idx], dtype=torch.float32)
        y_class = torch.tensor(self.classes[idx], dtype=torch.long)
        return input_ids.squeeze(0), attention_mask.squeeze(0), y_dist, y_class

    @staticmethod
    def augment(seq, delete_prob=0.02, substitute_prob=0.03, insert_prob=0.01):
        aas = "ACDEFGHIKLMNPQRSTVWY"
        seq = [aa for aa in seq if random.random() > delete_prob]
        seq = [random.choice(aas) if random.random() < substitute_prob else aa for aa in seq]
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
        sequence = self.sequences[idx]
        encoding = self.tokenizer(
            sequence,
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
            for line in file:
                if not line.startswith(">"):
                    sequences.append(line.strip())
    return sequences


class CustomDataCollator(DataCollatorForLanguageModeling):

    def __call__(self, features):
        batch = super().__call__(features)
        probability_matrix = batch["input_ids"]
        special_tokens_mask = batch["attention_mask"] == 0
        batch["input_ids"] = probability_matrix.masked_fill(special_tokens_mask, value=0.0)
        return batch
