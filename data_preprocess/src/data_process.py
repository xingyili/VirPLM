from src.HI import * 
from src.HA import *
from src.utils import *
import joblib
import tqdm
import numpy as np
def get_HI_data(subtype):
 
    HI_dir = "./data/" + subtype + "/HI"

    HI_file_paths = get_file_paths(HI_dir)

    
    virus_name_pattern = re.compile(r"(.*A/.*)")

    HI_data = HI()
    for file_path in tqdm.tqdm(HI_file_paths):
        HI_file = pd.ExcelFile(file_path, engine="openpyxl")
        for sheet_name in HI_file.sheet_names:
           
            origin_HI_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1, engine="openpyxl").fillna("")
            
            del_row_idx = []
            for row_idx in range(4, origin_HI_df.shape[0]):
                if not re.match(virus_name_pattern, str(origin_HI_df.iloc[row_idx, 0])):
                    del_row_idx.append(row_idx)
                  
                    if str(origin_HI_df.iloc[row_idx, 0]) not in ["REFERENCE VIRUSES", "TEST VIRUSES"]:
                        logger.warning("filename: {}, sheet: {}, {}".format(file_path, sheet_name, str(origin_HI_df.iloc[row_idx, 0])))
            df_fited_row = origin_HI_df.drop(del_row_idx).reset_index(drop=True)
            df_fited = df_fited_row.astype(str)
            try:
                df_fited.iloc[1,0], df_fited.iloc[2,0], df_fited.iloc[3,0] = "Passage history", "Ferret number", "Genetic group"
                df_fited.iloc[0] = ["Viruses", "Other information", "Genetic group", "Collection date", "Passage history"] + list(df_fited.iloc[0, 5:])
                df_fited.iloc[1:4, 1:5] = ""
                df_fited = df_fited.drop(columns=[df_fited.columns[1]])
                
            except Exception as e:
                print(f"-------------------------------------------------------------------发生异常: {e}")
            HI_data.add_table(subtype, file_path, sheet_name, df_fited)
            
    return HI_data

def get_HA_df(subtype):
    HA_file_path = "./data/" + subtype + "/HA/H3N2.fasta"
    logger.info("-------------------------- HA -------------------------")
    origin_HA_df = pd.read_csv(HA_file_path, delimiter='\t', skiprows=0, header=None, dtype=str)
    HA_odd_rows = origin_HA_df.iloc[::2].reset_index(drop=True)
    HA_even_rows = origin_HA_df.iloc[1::2].reset_index(drop=True)
    HA_df = pd.concat([HA_odd_rows, HA_even_rows], axis=1)
    HA_df.columns = ['virus_name', 'HA_seq']
    HA_df = get_processed_HA(HA_df)
    return HA_df