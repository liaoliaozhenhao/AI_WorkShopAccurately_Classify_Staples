#读写文件
import os
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def read_excel(path: str,header:int=0) -> pd.DataFrame:
    df=pd.read_excel(path,header=header, engine='openpyxl')
    print(f"{path}:len={len(df)}")
    return df

def write_excel_sheets(out_path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default
