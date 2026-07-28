# %% 
import pandas as pd
import tqdm
import os
import numpy as np
from src.HI import * 
from src.HA import *
from src.utils import *
from loguru import logger
import joblib
from src.data_process import *
from src.match import *
import argparse

logger.remove()  
logger.add("log/match.log", level="DEBUG", mode="w", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")  
subtype = 'H3N2'

 
parser = argparse.ArgumentParser(description="示例命令行参数程序")
parser.add_argument("--subtype", type=str, default="H3N2", help="数据亚型")
args = parser.parse_args()
subtype = args.subtype

if os.path.isfile("joblib/" + subtype+"HI_data.joblib"):
    HI_data = joblib.load("joblib/" + subtype+"HI_data.joblib")
else:
    HI_data = get_HI_data(subtype)
    joblib.dump(HI_data, "joblib/" + subtype+"HI_data.joblib")
HA_df = get_HA_df(subtype)

 
print("----- 合并HI表格 -----")
NHT_data_vector, AHT_data_vector  = get_HI_data_vector(HI_data)

 
print("----- 重复条目取平均 -----")
NHT_data_vector, AHT_data_vector = remove_duplicates(NHT_data_vector, AHT_data_vector)

 
print("----- 匹配 -----")
print(len(NHT_data_vector))
NHT_HA, NHT_HI = get_matched_data(HA_df, NHT_data_vector)
AHT_HA, AHT_HI = get_matched_data(HA_df, AHT_data_vector)
 
NHT_HA.to_csv("tmp/" + subtype + "_NHT_HA.csv", index=False)
NHT_HI.to_csv("tmp/" + subtype + "_NHT_HI.csv", index=False)

AHT_HA.to_csv("tmp/" + subtype + "_AHT_HA.csv", index=False)
AHT_HI.to_csv("tmp/" + subtype + "_AHT_HI.csv", index=False)

 
with open('tmp/' + subtype + '.fasta', 'w') as f:
    for idx, row in HA_df.iterrows():
        f.write(f">{row['isolate_name']}\n{row['seq']}\n")