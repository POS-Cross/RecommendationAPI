from typing import Optional
import json
from fastapi import FastAPI
from fpgrowth import DataItems
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

app = FastAPI()



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/maxlen/{max_len}/minsupport/{min_support}/itemset/{itemset}")
def read_data(max_len: int,min_support:float, itemset:str):
    #1. read dataset
    df_items = DataItems.read_dataset()
    print(df_items.head())
    #2. clean dataset
    df_items =DataItems.clean_data(df_items)

    #3. filter_data
    #itemset  =[142,21718,75254,77885,1292,21524,4166,68248]
    #itemset =df_items['dItemInternalKey'].unique().tolist()
    itemset = itemset.split(',')
    print(itemset)
    df_filtered = DataItems.filter_data(itemset ,df_items)

    #4. get products
    products=DataItems.get_products(df_filtered)

    #5. convert_to_listoflists
    orders =DataItems.convert_to_listoflists(df_items)

    #6. convert to 1 hot encoding
    orders_1hot =DataItems.get_1hot_encoding(orders)

    #7. call fpgrowth 
    
    fp =fpgrowth(orders_1hot, min_support=min_support, max_len=max_len, use_colnames=True)
    print(fp)
    #8. get assosciation rules:
    rules =association_rules(fp, metric="lift", min_threshold=10)
    print(rules)
    return rules.to_dict()





