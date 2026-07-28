import pandas as pd
from src.utils import *
from loguru import logger
from unidecode import unidecode
import tqdm


def get_processed_HA(HA_seq_df):
    ph_type_dict = get_ph_type_dict()
    data_dict = {
        "virus_name": [],
        "virus_location": [],
        "virus_id": [],
        "virus_year": [],
        "virus_ph": [],
        "seq": [],
    }
    cnt = 0
    re_tool = ReTool()
    for row_idx in range(len(HA_seq_df)):
        if invalid_gap(HA_seq_df.iloc[row_idx, 1]):
            cnt += 1
            continue
        try:
            virus_name = unidecode(HA_seq_df.iloc[row_idx, 0])
            virus_location = re_tool.get_virus_info(virus_name, 1)
            virus_id = re_tool.get_virus_info(virus_name, 2)
            virus_year = re_tool.get_virus_info(virus_name, 3)
            virus_ph = normalize_passage(re_tool.get_virus_info(virus_name, 5))
        except Exception as e:
            logger.error(str(e))
            continue

        data_dict["virus_name"].append(virus_name)
        data_dict["virus_location"].append(virus_location)
        data_dict["virus_id"].append(virus_id)
        data_dict["virus_year"].append(virus_year)
        data_dict["virus_ph"].append(virus_ph)
        data_dict["seq"].append(HA_seq_df.iloc[row_idx, 1])

    print("---   {}".format(cnt))
    ret = pd.DataFrame(data_dict).fillna("")
    ret["virus_ph_type"] = ret["virus_ph"].map(ph_type_dict).fillna("unknown")

    
    identity_cols = ["virus_location", "virus_id", "virus_year", "virus_ph"]
    duplicate_rows = ret.duplicated(identity_cols, keep=False)
    if duplicate_rows.any():
        duplicate_groups = ret.loc[duplicate_rows].groupby(identity_cols, dropna=False)
        conflict_count = sum(group["seq"].nunique() > 1 for _, group in duplicate_groups)
        logger.info(
            "Collapsed {} duplicate HA rows by standardized strain + passage; "
            "{} duplicate groups contained more than one sequence and kept the first input record.".format(
                int(ret.duplicated(identity_cols, keep="first").sum()), conflict_count
            )
        )
    ret = ret.drop_duplicates(identity_cols, keep="first").reset_index(drop=True)
    ret["isolate_name"] = ret.apply(
        lambda row: isolate_name(
            row["virus_location"], row["virus_id"], row["virus_year"], row["virus_ph"]
        ),
        axis=1,
    )
    return ret


 
def invalid_gap(seq: str) -> bool:
    if "X" in seq or "B" in seq or "Z" in seq or "J" in seq:
        return True
    if seq[0] == "-" or seq[-1] == "-":
        return True
    if seq.count("-") * 10 > len(seq):
        return True
    return False
