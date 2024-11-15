import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ais_data_20210101_csv = pd. read_csv("AIS_2021_01_01.csv")
print("loading csv")
ais_data_20210101_csv.to_parquet("ais_data_20210101.parquet")