
#!pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import time

# 載入Excel數據
data = pd.read_excel('交易資料集.xlsx')

# 剔除數量為零或負值的交易記錄
data = data[data['QUANTITY'] > 0]

# 將交易數據轉換為物品列表
transformed_data = data.groupby('INVOICE_NO')['ITEM_ID'].apply(list).tolist()

# 使用TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transformed_data).transform(transformed_data)
df = pd.DataFrame(te_ary, columns=te.columns_)


需調整
# 設定支持度和信心度()
s = 0.001  # 支持度
c = 0.5   # 信心度

# 使用Apriori算法
frequent_itemsets_ap = apriori(df, min_support=s, use_colnames=True)
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=c)

# 使用FP-Growth算法
frequent_itemsets_fp = fpgrowth(df, min_support=s, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=c)


# 将前件和后件的 frozenset 转换为逗号分隔的字符串
rules_ap['antecedents'] = rules_ap['antecedents'].apply(lambda x: ', '.join(str(item) for item in x))
rules_ap['consequents'] = rules_ap['consequents'].apply(lambda x: ', '.join(str(item) for item in x))

rules_fp['antecedents'] = rules_fp['antecedents'].apply(lambda x: ', '.join(str(item) for item in x))
rules_fp['consequents'] = rules_fp['consequents'].apply(lambda x: ', '.join(str(item) for item in x))

# 现在将修改后的规则输出到 CSV 文件
rules_ap.to_csv('apriori_rules.csv', index=False)
rules_fp.to_csv('fpgrowth_rules.csv', index=False)

# 讀取規則
rules_ap_loaded = pd.read_csv('apriori_rules.csv')
rules_fp_loaded = pd.read_csv('fpgrowth_rules.csv')

def recommend_products(rules, products, max_recommendations=5):
    """
    根據關聯規則推薦產品。
    """
    recommendations = set()
    for product in products:
        product_str = str(product)
        # 僅匹配前件恰好等於單一產品的規則
        matched_rules = rules[rules['antecedents'] == product_str]
        for idx, row in matched_rules.iterrows():
            # 後件是逗號分隔的字串，需要轉換為整數列表
            consequents = set(int(item.strip()) for item in row['consequents'].split(','))
            recommendations.update(consequents)
            if len(recommendations) >= max_recommendations:
                return list(recommendations)
    return list(recommendations)
# 假設用戶輸入的產品列表
user_products = ['1697, 70509','1697']
recommended_products_AP = recommend_products(rules_ap, user_products)
recommended_products_FP = recommend_products(rules_fp, user_products)

# 打印推薦結果
print("AP關聯規則推薦產品:", recommended_products_AP)
print("FP關聯規則推薦產品:", recommended_products_FP)

# 測量Apriori算法的時間
start_time = time.time()
apriori(df, min_support=s, use_colnames=True)  
ap_time = time.time() - start_time

# 測量FP-Growth算法的時間
start_time = time.time()
fpgrowth(df, min_support=s, use_colnames=True)
fp_time = time.time() - start_time

print(f"Apriori Time: {ap_time}")
print(f"FP-Growth Time: {fp_time}")
