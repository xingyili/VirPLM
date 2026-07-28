import pandas as pd
from src.utils import *
from loguru import logger
from unidecode import unidecode
import numpy as np
import tqdm

class HI:
    def __init__(self):
        self.tables = []

    def add_table(self, subtype, file_path, sheet_name, df_fited):
        splitted_tables = HI.get_splitted_tables(subtype, file_path, sheet_name, df_fited)
        self.tables.append(HI.merge_table(splitted_tables))
    
    @classmethod
    def merge_table(cls, table_list):
        merged_ret = pd.DataFrame(columns=table_list[0].data_vector.columns)
        for table in table_list:
            if merged_ret.empty:
                merged_ret = table.data_vector
            else:
                merged_ret = pd.concat([merged_ret, table.data_vector],axis=0)
        NHT_data_vector = merged_ret.copy(deep=True)
        NHT_data_vector = NHT_data_vector[NHT_data_vector['NHT_distance'] != -1].drop(columns = ['AHT_distance', 'AHT_class'])
        AHT_data_vector = merged_ret.copy(deep=True)
        AHT_data_vector = AHT_data_vector[AHT_data_vector['AHT_distance'] != -1].drop(columns = ['NHT_distance', 'NHT_class'])
        
        NHT_data_vector = NHT_data_vector.rename(columns={'NHT_distance': 'distance', 'NHT_class': 'class'})
        AHT_data_vector = AHT_data_vector.rename(columns={'AHT_distance': 'distance', 'AHT_class': 'class'})
        
        ret = table_list[0]
        ret.NHT_data_vector, ret.AHT_data_vector = remove_duplicates(NHT_data_vector,AHT_data_vector)
        ret.homo_titers = None
        ret.data_matix = None
        return ret


    @classmethod
    def get_splitted_tables(cls, subtype, file_path, sheet_name, data_matix):
        data_matix = data_matix.reset_index(drop=True)
        sr_set = set()
        for col_idx in range(4, data_matix.shape[1]):
            sr_set.add((unidecode(data_matix.iloc[0, col_idx]), data_matix.iloc[1, col_idx]))
   
        for col_idx1 in range(4, data_matix.shape[1]):
            for col_idx2 in range(col_idx1 + 1, data_matix.shape[1]):
                sr1_name = unidecode(data_matix.iloc[0, col_idx1])
                sr1_ph = data_matix.iloc[1, col_idx1]
                sr2_name = unidecode(data_matix.iloc[0, col_idx2])
                sr2_ph = data_matix.iloc[1, col_idx2]
                if (sr1_name, sr1_ph) == (sr2_name, sr2_ph):
                    df1 = data_matix.drop(columns = [data_matix.columns[col_idx2]])
                    df2 = data_matix.drop(columns = [data_matix.columns[col_idx1]])
                    return HI.get_splitted_tables(subtype, file_path, sheet_name, df1) + HI.get_splitted_tables(subtype, file_path, sheet_name, df2)
 
        for row_idx1 in range(4, data_matix.shape[0]):
            for row_idx2 in range(row_idx1 + 1, data_matix.shape[0]):
                at1_name = unidecode(data_matix.iloc[row_idx1, 0])
                at1_ph = data_matix.iloc[row_idx1, 1]
                at2_name = unidecode(data_matix.iloc[row_idx2, 0])
                at2_ph = data_matix.iloc[row_idx2, 1]
                
                if (at1_name, at1_ph) == (at2_name, at2_ph):

                    if (at1_name, at1_ph) in sr_set:
                        
                        
                        df1 = data_matix.drop(index=row_idx2)
                        df2 = data_matix.drop(index=row_idx1)
                        return HI.get_splitted_tables(subtype, file_path, sheet_name, df1) + HI.get_splitted_tables(subtype, file_path, sheet_name, df2)
        return [HI_table(subtype, file_path, sheet_name, data_matix)]

class HI_table:
    def __init__(self, subtype, file_path, sheet, data_matix):
        self.max_titer = 0
        self.subtype = subtype
        self.file_path = file_path
        self.sheet = sheet
        self.data_matix = data_matix
        self.re_tool = ReTool()
        self.data_vector = self.get_data_vector()
        self.homo_titers = {} 
        self.AHT_data_vector = None
        self.NHT_data_vector = None

        self.update_homo_titers()  
        self.update_NHT()  
        self.update_AHT()  
        



    def update_homo_titers(self):
        tmp_df = self.data_vector
        sr_set  = set(self.data_vector[['sr_location', 'sr_id', 'sr_year', 'sr_ph']].apply(tuple, axis=1).tolist())

        for sr in sr_set:
            homo_df = tmp_df[(tmp_df['at_location'] == sr[0]) & (tmp_df['at_id'] == sr[1]) & (tmp_df['at_year'] == sr[2]) & (tmp_df['at_ph'] == sr[3]) & (tmp_df['sr_location'] == sr[0]) & (tmp_df['sr_id'] == sr[1]) & (tmp_df['sr_year'] == sr[2]) & (tmp_df['sr_ph'] == sr[3])]
            if len(homo_df) == 0:
                logger.warning("filename: {}, sheet: {}, no homo: {}".format(self.file_path, self.sheet, str(sr)))
            elif len(homo_df) > 1:
                logger.warning("filename: {}, sheet: {}, more than 1 homo: {}".format(self.file_path, self.sheet, str(sr)))
            else:
                cur_titer = homo_df["titer"].iloc[0]
                if not cur_titer > 0:
                    logger.error(" filepath: {} sheet: {} , homo titer is lack!!!")
                self.homo_titers[sr] = cur_titer
        pass 

    def update_NHT(self):
        for idx in range(len(self.data_vector)):
            sr_key = (self.data_vector["sr_location"].iloc[idx], self.data_vector["sr_id"].iloc[idx], self.data_vector["sr_year"].iloc[idx], self.data_vector["sr_ph"].iloc[idx])
            cur_titer = self.data_vector["titer"].iloc[idx]
            if cur_titer < 0:
                continue
            assert(cur_titer!=0)
            if sr_key in self.homo_titers.keys():
                if self.data_vector["at_location"].iloc[idx] == self.data_vector["sr_location"].iloc[idx] and self.data_vector["at_id"].iloc[idx] == self.data_vector["sr_id"].iloc[idx] and self.data_vector["at_year"].iloc[idx] == self.data_vector["sr_year"].iloc[idx] and self.data_vector["at_ph"].iloc[idx] != self.data_vector["sr_ph"].iloc[idx]: # H1 2018-09 A/michigan/272/2017这个毒株
                    continue
                homo_titer = float(self.homo_titers[sr_key])
            else:
                continue
 
            distance = np.log2(np.maximum(homo_titer, cur_titer) / cur_titer)
            self.data_vector['NHT_distance'].iat[idx] = distance
      
            self.data_vector['NHT_class'].iat[idx] = 0 if distance < 2.0 else 1  

    def update_AHT(self):
        
        homo_keys = list(self.homo_titers.keys())
        for i in range(len(homo_keys)):
    
            for j in range(i, len(homo_keys)):  
                at_key, sr_key = homo_keys[i], homo_keys[j]
 
                if at_key[0] == sr_key[0] and at_key[1] == sr_key[1] and at_key[2] == sr_key[2] and at_key[3] != sr_key[3]:
                    continue
                Hii = self.get_titer_by_key(at_key, at_key)
                Hjj = self.get_titer_by_key(sr_key, sr_key)
                Hij = self.get_titer_by_key(at_key, sr_key)
                Hji = self.get_titer_by_key(sr_key, at_key)
                if Hii <= 0 or Hjj <= 0 or Hij <= 0 or Hji <= 0:
                    continue
 
                if at_key > sr_key:
                    tmp = at_key
                    at_key = sr_key
                    sr_key = tmp
                dist_series = self.data_vector.query("at_location == @at_key[0] and at_id == @at_key[1] and at_year == @at_key[2] and at_ph == @at_key[3] and sr_location == @sr_key[0] and sr_id == @sr_key[1] and sr_year == @sr_key[2] and sr_ph == @sr_key[3]")
                distance = np.sqrt((Hii*Hjj) / (Hij*Hji))
                self.data_vector['AHT_distance'].iat[dist_series.index[0]] = distance
                
                self.data_vector['AHT_class'].iat[dist_series.index[0]] = 0 if distance < 4.0 else 1
        
    def get_titer_by_key(self, at_key, sr_key):
        ret = self.data_vector.query("at_location == @at_key[0] and at_id == @at_key[1] and at_year == @at_key[2] and at_ph == @at_key[3] and sr_location == @sr_key[0] and sr_id == @sr_key[1] and sr_year == @sr_key[2] and sr_ph == @sr_key[3]")
        assert(len(ret)==1)
        return ret["titer"].iloc[0]

    def get_data_vector(self):
        ph_type_dict  = get_ph_type_dict()
        data_dict = {
            "at_name":[], 
            "at_location":[], 
            "at_id":[], 
            "at_year":[], 
            "at_ph":[], 
            "at_gg":[], 
            "at_index": [], 
            "sr_name":[], 
            "sr_location":[], 
            "sr_id":[], 
            "sr_year":[], 
            "sr_ph":[], 
            "sr_gg":[], 
            "sr_index": [], 
            "max_year": [],
            "min_year": [],
            "titer":[], 
            "titer_type":[], 
            "NHT_class": [], 
            "NHT_distance": [], 
            "AHT_class": [], 
            "AHT_distance": [] 
            }
        errors = set() 
        for row_idx in range(4, len(self.data_matix)):
            try:
                at_name = unidecode(self.data_matix.iloc[row_idx, 0])
                at_location = self.re_tool.get_virus_info(at_name, 1)
                at_id = self.re_tool.get_virus_info(at_name, 2)
                at_year = self.re_tool.get_virus_info(at_name, 3)
                at_ph = self.data_matix.iloc[row_idx, 3]
                at_gg = self.data_matix.iloc[row_idx, 1]
            except Exception as e:
                if str(e) not in errors:
                    logger.error(str(e))
                    logger.info("--- " + self.file_path + " --- sheet: " + self.sheet + " --- ")
                    errors.add(str(e))
                continue
                 
            at_ph = normalize_passage(at_ph)
            at_gg = ''.join(at_gg.lower().split())
            for col_idx in range(4, len(self.data_matix.columns)):
                
                try:
                    sr_name = unidecode(self.data_matix.iloc[0, col_idx])
                    sr_location = self.re_tool.get_virus_info(sr_name, 1)
                    sr_id = self.re_tool.get_virus_info(sr_name, 2)
                    sr_year = self.re_tool.get_virus_info(sr_name, 3)
                    sr_ph = self.data_matix.iloc[1, col_idx]
                    sr_gg = self.data_matix.iloc[3, col_idx]
                except Exception as e:
                    if str(e) not in errors:
                        logger.error(str(e))
                        logger.info("--- " + self.file_path + " --- sheet: " + self.sheet + " --- ")
                        errors.add(str(e))
                    continue
                     

                sr_ph = normalize_passage(sr_ph)
                sr_gg = ''.join(sr_gg.lower().split())
                data_dict["at_name"].append(at_name)
                data_dict["at_location"].append(at_location)
                data_dict["at_id"].append(at_id)
                data_dict["at_year"].append(at_year)
                data_dict["at_ph"].append(at_ph)
                data_dict["at_gg"].append(at_gg)
                data_dict["at_index"].append("")
               

                data_dict["sr_name"].append(sr_name)
                data_dict["sr_location"].append(sr_location)
                data_dict["sr_id"].append(sr_id)
                data_dict["sr_year"].append(sr_year)
                data_dict["sr_ph"].append(sr_ph)
                data_dict["sr_gg"].append(sr_gg)
                data_dict["sr_index"].append("")
                

                titer, titer_type = self.get_titer_value(self.data_matix.iloc[row_idx, col_idx])
                data_dict["max_year"].append(max(at_year, sr_year))
                data_dict["min_year"].append(min(at_year, sr_year))
                data_dict["titer"].append(titer)
                self.max_titer = self.max_titer if self.max_titer > titer else titer
                if titer==0:
                    pass
                data_dict["titer_type"].append(titer_type)
                
                data_dict["NHT_class"].append(-1)
                data_dict["NHT_distance"].append(-1.0)
                data_dict["AHT_class"].append(-1)
                data_dict["AHT_distance"].append(-1.0)
        ret = pd.DataFrame(data_dict).fillna("")
        ret['at_ph_type'] = ret['at_ph'].map(ph_type_dict)
        ret['sr_ph_type'] = ret['sr_ph'].map(ph_type_dict)
        for i in range(len(ret)):
            if pd.isna(ret['at_ph_type'].iloc[i]):
                logger.warning(" ------------has no map ph!!!!".format(ret['at_ph'].iloc[i]))
            if pd.isna(ret['sr_ph_type']).iloc[i]:
                logger.warning("{}------------has no map ph!!!!".format(ret['sr_ph'].iloc[i]))
        ret["at_ph_type"] = ret["at_ph_type"].fillna('unknown')
        ret["sr_ph_type"] = ret["sr_ph_type"].fillna('unknown')
        ret = ret[['at_name', 'at_location', 'at_id', 'at_year', 'at_ph', 'at_gg', 'at_ph_type', 'sr_name', 'sr_location', 'sr_id', 'sr_year', 'sr_ph', 'sr_gg', 'sr_ph_type', 'max_year', 'min_year', 'titer', 'NHT_class', 'NHT_distance', 'AHT_class', 'AHT_distance']]
        return ret
    
    def get_titer_value(self, titer_str) -> float:
         
        if is_number(titer_str):
            titer, titer_type = float(titer_str), "NORMAL"
        elif len(titer_str) > 1 and titer_str[0] == ">":
            if is_number(titer_str[1:]):
   
                titer, titer_type = float(titer_str[1:]) * 2, "MORE"
            else:
                titer, titer_type = -1e6, "LACK"
        elif len(titer_str) > 1 and titer_str[0] == "<":
            
            if is_number(titer_str[1:]):
                titer, titer_type = -1e6, "LESS"
               
            else:
                titer, titer_type = -1e6, "LACK"
        else:
            titer, titer_type = -1e6, "LACK"
        if titer == 0:
            return -1e6, "LACK" 
        else:
            return titer, titer_type
 
def find_ph_loc(HI_tables, ph_item):
    ret = set()
    for table in HI_tables:
        tmp_df = table.data_vector
        for row_idx in range(len(tmp_df)):
            at_ph = tmp_df.iloc[row_idx]['at_ph']
            sr_ph = tmp_df.iloc[row_idx]['sr_ph']
            if at_ph == ph_item or sr_ph == ph_item:
                if (table.file_path, table.sheet) not in ret:
                    ret.add((table.file_path, table.sheet))
    return ret

def get_HI_data_vector(HI_data):
    
    NHT_ret = pd.DataFrame(columns=HI_data.tables[0].NHT_data_vector.columns)
    AHT_ret = pd.DataFrame(columns=HI_data.tables[0].AHT_data_vector.columns)
    for table in tqdm.tqdm(HI_data.tables):
        if NHT_ret.empty:
            NHT_ret = table.NHT_data_vector
        else:
            NHT_ret = pd.concat([NHT_ret, table.NHT_data_vector],axis=0)

        if AHT_ret.empty:
            AHT_ret = table.AHT_data_vector
        else:
            AHT_ret = pd.concat([AHT_ret, table.AHT_data_vector],axis=0)
    
    return NHT_ret, AHT_ret

def remove_duplicates_and_spilt(HI_data_vector):
    NHT_data_vector = HI_data_vector.copy(deep=True)
    NHT_data_vector = NHT_data_vector[NHT_data_vector['NHT_distance'] != -1].drop(columns = ['AHT_distance', 'AHT_class'])
    AHT_data_vector = HI_data_vector.copy(deep=True)
    AHT_data_vector = AHT_data_vector[AHT_data_vector['AHT_distance'] != -1].drop(columns = ['NHT_distance', 'NHT_class'])

    NHT_columns_to_check = ['at_location', 'at_id', 'at_year', 'sr_location', 'sr_id', 'sr_year']
    NHT_column_to_average = "NHT_distance"
 
    NHT_merged = NHT_data_vector.groupby(NHT_columns_to_check, as_index=False).agg({NHT_column_to_average: "mean", 'at_name':'first', 'sr_name':'first', 'max_year': 'first', 'min_year': 'first', 'NHT_class': 'first', 'at_ph_type': lambda x: "" if len(x) > 1 else x.iloc[0], 'sr_ph_type': lambda x: "" if len(x) > 1 else x.iloc[0]})
 
    AHT_columns_to_check = ['at_location', 'at_id', 'at_year', 'sr_location', 'sr_id', 'sr_year']
    AHT_column_to_average = "AHT_distance"
    AHT_merged = AHT_data_vector.groupby(AHT_columns_to_check, as_index=False).agg({AHT_column_to_average: "mean", 'at_name':'first', 'sr_name':'first', 'max_year': 'first', 'min_year': 'first', 'AHT_class': 'first', 'at_ph_type': lambda x: "" if len(x) > 1 else x.iloc[0], 'sr_ph_type': lambda x: "" if len(x) > 1 else x.iloc[0]})

 
    NHT_merged.loc[NHT_merged["NHT_distance"] >= 2.0, "NHT_class"] = 1
    NHT_merged.loc[NHT_merged["NHT_distance"] < 2.0, "NHT_class"] = 0
    AHT_merged.loc[AHT_merged["AHT_distance"] >= 4.0, "AHT_class"] = 1
    AHT_merged.loc[AHT_merged["AHT_distance"] < 4.0, "AHT_class"] = 0

    return NHT_merged, AHT_merged

def remove_duplicates(NHT_data_vector, AHT_data_vector):

    NHT_columns_to_check = ['at_location', 'at_id', 'at_year', 'at_ph', 'sr_location', 'sr_id', 'sr_year', 'sr_ph']
    NHT_column_to_average = "distance"
    NHT_merged = NHT_data_vector.groupby(NHT_columns_to_check, as_index=False, dropna=False).agg({NHT_column_to_average: "mean", 'at_name':'first', 'sr_name':'first', 'max_year': 'first', 'min_year': 'first', 'class': 'first', 'at_ph_type': 'first', 'sr_ph_type': 'first'})
 
    AHT_columns_to_check = ['at_location', 'at_id', 'at_year', 'at_ph', 'sr_location', 'sr_id', 'sr_year', 'sr_ph']
    AHT_column_to_average = "distance"
    AHT_merged = AHT_data_vector.groupby(AHT_columns_to_check, as_index=False, dropna=False).agg({AHT_column_to_average: "mean", 'at_name':'first', 'sr_name':'first', 'max_year': 'first', 'min_year': 'first', 'class': 'first', 'at_ph_type': 'first', 'sr_ph_type': 'first'})

   
    NHT_merged.loc[NHT_merged["distance"] >= 2.0, "class"] = 1
    NHT_merged.loc[NHT_merged["distance"] < 2.0, "class"] = 0
    AHT_merged.loc[AHT_merged["distance"] >= 4.0, "class"] = 1
    AHT_merged.loc[AHT_merged["distance"] < 4.0, "class"] = 0
     
    return NHT_merged, AHT_merged