import os
import re
import pandas as pd
from datetime import datetime
def get_file_paths(dir):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[0] == '.':
                continue 
            absolute_path = os.path.join(root, file)
            file_paths.append(absolute_path)
    return file_paths

def is_number(s):
    try:
        float(s)   
        return True
    except ValueError:
        return False

class ReTool:
    def __init__(self):
        
        self.virus_name_pattern = re.compile(r'.*([A])/+([A-Za-z_\-\.\'\s&]+)/+([A-Za-z_\-\.\'\d\s/\(\)]+)/+([\d_\-\s]+)(\|+(.*))?')
        self.no_digit_pattern = re.compile(r'\D')
        self.ph_pattern = re.compile(r'.*\|(.*)')
    
    def get_virus_info(self, virus_name, loc):
        match = re.match(self.virus_name_pattern, virus_name)
        if match:
            extracted = match.groups()
            ret = extracted[loc] if loc != 3 else self.align_year(virus_name, extracted[loc])
            return ''.join(ret.lower().split()) if ret != None else ""
        else:
            raise Exception(virus_name + " has virus name errrr!!!")
        
    def get_ph_info(self, virus_name):
        match = re.match(self.ph_pattern, virus_name)
        if match:
            extracted = match.groups()
            ret = extracted[0] if extracted[0] !=None else ""
            return ''.join(ret.lower().split()) if ret != None else ""
        else: 
            raise Exception(virus_name + "'s ph information is errr!!!")
    def align_year(self, virus_name, origin_year):
        origin_year = re.sub(self.no_digit_pattern, "", origin_year)
        if len(origin_year) == 3:
            raise Exception(virus_name + "'s year is " + origin_year + ", which is not correct!!!")
        if len(origin_year) == 2:
            origin_year = ("19" if int("20" + origin_year) > datetime.now().year else "20") + origin_year
        return origin_year
    
def get_ph_type_dict():
    ph_type_df = pd.read_excel("data_for_ph/all_ph.xlsx", sheet_name="com", header=0).fillna("")
    return dict(zip(ph_type_df['ph'], ph_type_df['ph_type']))

def normalize_passage(passage) -> str:
     
    if passage is None or pd.isna(passage):
        return ""
    return "".join(str(passage).lower().split())


def standardized_strain_name(location, virus_id, year) -> str:
    
    return "A/{}/{}/{}".format(str(location).lower(), str(virus_id).lower(), str(year))


def isolate_name(location, virus_id, year, passage) -> str:
    
    normalized = normalize_passage(passage)
    return "{}|{}".format(
        standardized_strain_name(location, virus_id, year),
        normalized if normalized else "unknown",
    )
