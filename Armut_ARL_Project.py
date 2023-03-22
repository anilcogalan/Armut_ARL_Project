import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


df_ = pd.read_csv('armut_data.csv')
df = df_.copy()

df.head()
df.info()

# ServiceID represents a different service for each CategoryID.
df['Service'] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

# Firstly, need to convert the create_date variable to datetime type.
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['New_Date'] = df['CreateDate'].dt.strftime("%Y-%m")

# Create a new category that merging UserID & New_Date
df ['BasketID'] = [str(row[0])+ "_" + str(row[5]) for row in df.values]

def arl_preparation_df (dataframe = pd.DataFrame):
    """
    Fill in the cells with the information not received in DataFrame.
    Args:
        dataframe
    Returns:
        Dataframe: DataFrame
    """
    return dataframe. \
        groupby(['BasketID', 'Service'])['Service'].count(). \
        unstack(). \
        fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)

arl_preparation_df = arl_preparation_df (df)

def create_rules(dataframe):
    """
    The Apriori algorithm prepares the table to create the association rule.
    It asks for min_support.
    Rules calculate lift,confidence, support values.
    antecedents = X consequents = Y

    Args:
        dataframe

    Returns:
        rules: DataFrame

    """
    dataframe = arl_preparation_df(dataframe)
    frequent_items = apriori(dataframe,
                             min_support=0.01,
                             use_colnames=True)

    rules = association_rules(frequent_items,
                              metric='support',
                              min_threshold=0.01)
    return rules

rules = create_rules(df)

def arl_recommender(rules_df, product_id, rec_count=1):
    """
        It creates a recommendation system according to the lift parameter.
    Args:
        rules_df: DataFrame that  created the association rule
        product_id: Service Type
        rec_count: Number of Recommendations

    Returns:
        recommendation_list : List

    """
    sorted_rules = rules_df.sort_values("lift", ascending = False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]['consequents'])[0])
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[0:rec_count]

# Example
arl_recommender(rules,"2_0",5)
