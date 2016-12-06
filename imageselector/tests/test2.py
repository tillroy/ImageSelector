# coding: utf-8
from operator import itemgetter
import random

import numpy as np
import pandas as pd


data = list()
random.seed(80)
for el in range(10):

    res = {
        "row": random.random(),
        "col": random.random(),
        "type": random.choice(range(2)),
    }
    data.append(res)

df = pd.DataFrame(data, columns=("row", "col", "type"))
df["group"] = (df["type"].diff() != 0).cumsum()

# get groups for further processing
df = df.set_index(["group", "type"])
# res = sorted([el for el in df.groupby(["group", "type"]).groups], key=itemgetter(0))
res = sorted([el for el in df.groupby(by=[df.index.get_level_values(0), df.index.get_level_values(1)]).groups],
             key=itemgetter(0))
if res[0][1] != 1:
    del res[0]

# df = df.set_index(["group", "type"])
print(df)
# take transitions as a start of object and fill as a body of an object
for start_group_ind, end_group_ind in zip(res[::2], res[1::2]):
    start_group = df.ix[start_group_ind[0], :]
    end_group = df.ix[end_group_ind[0], :]

    start_point = start_group.tail(1)
    end_point = end_group.tail(1)

    sample_body = start_point.append(end_group)
    body_coords = [el for el in sample_body[["row", "col"]].values]
    print(body_coords)


