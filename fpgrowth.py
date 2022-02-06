import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import association_rules

class DataItems:
    #CSV = "http://ec2-3-128-207-73.us-east-2.compute.amazonaws.com:8002/uploads/uploads/2021/9/20/19e5badc-19f3-11ec-b8c6-09d958cd918a.csv"
    CSV='query_sample.csv'
    ''' read the dataset'''
    @classmethod
    def read_dataset(cls):
        df_items=pd.read_csv(DataItems.CSV)

        return df_items

    ''' clean the dataframe by dropping na'''
    @staticmethod
    def clean_data(df):
        df.dropna()
        return df 

    """ filter the dataframe based on list of items """
    @staticmethod
    def filter_data(items,df):
        df[df['dItemInternalKey'].isin(items)]
        return df 

    @staticmethod
    def get_products( df):
        products = df[['dItemInternalKey', 'ItemName']]
        products = products[~products.duplicated()]
        # Set the index to StockCode
        products = products.set_index('dItemInternalKey')

        # Convert to Series for eve easier lookups
        products = products['ItemName']
        return products        

    @staticmethod
    def convert_to_listoflists(df_items):     
        orders=df_items[['dTicketInternalKey','dItemInternalKey']]
        orders = orders.groupby('dTicketInternalKey')['dItemInternalKey'].apply(list).reset_index()
        return orders 

    @staticmethod
    def get_1hot_encoding(df):
        te = TransactionEncoder()
        te.fit(df['dItemInternalKey'])
        orders_1hot = te.transform(df['dItemInternalKey'])
        orders_1hot = pd.DataFrame(orders_1hot, columns =te.columns_)
        return orders_1hot   

    @staticmethod
    def predict(antecedent, rules, max_results= 6):
        
        # get the rules for this antecedent
        preds = rules[rules['antecedents'] == antecedent]
        
        # a messy way to convert a frozen set with one element to string
        preds = preds['consequents'].apply(iter).apply(next)
        
        return preds[:max_results].reset_index(drop=True)


    @staticmethod
    def get_association_rules(df):
        rules = association_rules(df, metric="lift", min_threshold=10)
        return rules

