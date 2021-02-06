import pandas as pd
import numpy as np
import math

file = pd.read_csv('vocabulary.csv', sep=',')
ids = file['Index'].values  # previous label id
v1 = file['Vertical1'].values  # categories belong

# count all categories
category = []
for v in v1:
    if str(v) != 'nan' and not (v in category):
        category.append(v)

cates = len(category)  # num of categories,25 in fact
print(cates)

# list for each category
group = np.ndarray(cates, dtype=list)
for l in range(cates):
    group[l] = []

# add label id to each category
for i in range(len(ids)):
    for j in range(cates):
        if v1[i] == category[j]:
            group[j].append(i)

# categories arranged one by one
new_to_prev = []
cate_len=[]
for j in range(cates):
    cate_len.append(len(group[j]))
    print(category[j], len(group[j]))

df = pd.DataFrame.from_dict({'Category': category, 'Label_num': cate_len})
df.to_csv('Category.csv', header=True, index=False, columns=['Category', 'Labels'])

#mapping tuple
tu = []
for i in range(3862):
    tu.append([new_to_prev[i], i])
tu = np.asarray(tu)

df = pd.DataFrame.from_dict({'OldLabel': tu[:, 0], 'NewLabel': tu[:, 1]})
df.to_csv('Mapping_out.csv', header=True, index=False, columns=['OldLabel', 'NewLabel'])

#sort according to old label
tu = sorted(tu, key=lambda x: x[0])

df = pd.DataFrame.from_dict({'OldLabel': tu[:, 0], 'NewLabel': tu[:, 1]})
df.to_csv('Mapping_in.csv', header=True, index=False, columns=['OldLabel', 'NewLabel'])
