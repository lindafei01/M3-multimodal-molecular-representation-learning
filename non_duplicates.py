import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('data/lnp.csv')

# 对 SMILES 列进行去重，只保留第一个
df.drop_duplicates(subset='smiles', keep='first', inplace=True)

# 保存去重后的结果到新的 CSV 文件
df.to_csv('data/lnp_unique.csv', index=False)