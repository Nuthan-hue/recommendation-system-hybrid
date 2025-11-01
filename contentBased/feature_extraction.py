import pandas as pd
from parso.python.tree import String

haeders_list = {"unnamed":int, "user_id": int, "stream_id": int , "streamer_name": str, "time_start": int, "time_stop": int}
df= pd.read_csv("/Volumes/SD_Card/hybrid_reccomendations/data/100k_a.csv", header= None)
df.columns= haeders_list
df.to_csv('/Volumes/SD_Card/hybrid_reccomendations/data/100k_a.csv', index=True)
