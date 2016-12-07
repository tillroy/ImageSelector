# coding: utf-8
from operator import itemgetter
import random

import numpy as np
import pandas as pd

data1 = list()
random.seed(80)
for el in range(10):

    res = {
        "row": random.random(),
        "col": random.random(),
    }
    data1.append(res)

df1 = pd.DataFrame(data1, columns=("row", "col"))


data2 = [{
    "row": 0.271491,
    "col": 0.541907,
}]


df2 = pd.DataFrame(data2, columns=("row", "col"))

print(df1)
print(df2)

print(df1.intersection(df2))
