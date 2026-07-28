import tqdm
import pandas as pd
from loguru import logger
from src.utils import isolate_name, normalize_passage


def _build_ha_lookup(HA_df):
    
    exact = {}
    by_strain = {}
    for _, row in HA_df.iterrows():
        strain_key = (
            row["virus_location"],
            row["virus_id"],
            row["virus_year"],
        )
        passage = normalize_passage(row["virus_ph"])
        exact_key = strain_key + (passage,)
        record = row.to_dict()
        exact[exact_key] = record
        by_strain.setdefault(strain_key, []).append(record)
    return exact, by_strain


def _resolve_ha_record(exact_lookup, strain_lookup, location, virus_id, year, passage):
     
    strain_key = (location, virus_id, year)
    passage = normalize_passage(passage)
    exact_key = strain_key + (passage,)
    if exact_key in exact_lookup:
        return exact_lookup[exact_key]
    candidates = strain_lookup.get(strain_key, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def get_matched_data(HA_df, data_vector):
    exact_lookup, strain_lookup = _build_ha_lookup(HA_df)
    matched_HA = {
        "index": [],
        "name": [],
        "location": [],
        "id": [],
        "year": [],
        "seq": [],
    }
    matched_HI = {
        "at_index": [],
        "sr_index": [],
        "max_year": [],
        "min_year": [],
        "distance": [],
        "class": [],
    }
    isolate_to_index = {}
    skipped_ambiguous = 0

    def add_isolate(record):
        name = isolate_name(
            record["virus_location"],
            record["virus_id"],
            record["virus_year"],
            record["virus_ph"],
        )
        if name not in isolate_to_index:
            idx = len(matched_HA["name"])
            isolate_to_index[name] = idx
            matched_HA["index"].append(idx)
            matched_HA["name"].append(name)
            matched_HA["location"].append(record["virus_location"])
            matched_HA["id"].append(record["virus_id"])
            matched_HA["year"].append(record["virus_year"])
            matched_HA["seq"].append(record["seq"])
        return isolate_to_index[name]

    for row_idx in tqdm.tqdm(range(len(data_vector))):
        row = data_vector.iloc[row_idx]
        at_record = _resolve_ha_record(
            exact_lookup,
            strain_lookup,
            row["at_location"],
            row["at_id"],
            row["at_year"],
            row.get("at_ph", ""),
        )
        sr_record = _resolve_ha_record(
            exact_lookup,
            strain_lookup,
            row["sr_location"],
            row["sr_id"],
            row["sr_year"],
            row.get("sr_ph", ""),
        )
        if at_record is None or sr_record is None:
            skipped_ambiguous += 1
            continue

        matched_HI["at_index"].append(add_isolate(at_record))
        matched_HI["sr_index"].append(add_isolate(sr_record))
        matched_HI["max_year"].append(row["max_year"])
        matched_HI["min_year"].append(row["min_year"])
        matched_HI["distance"].append(row["distance"])
        matched_HI["class"].append(row["class"])

    if skipped_ambiguous:
        logger.warning(
            "Skipped {} HI pairs because passage-specific HA resolution was ambiguous or missing.".format(
                skipped_ambiguous
            )
        )
    return pd.DataFrame(matched_HA), pd.DataFrame(matched_HI)
