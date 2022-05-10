import numpy as np
import pandas as pd

def done(name:str)->None:
    fe_boru=pd.read_csv(name)
    print(fe_boru.outcome.value_counts())
    print()
    print()
    print(fe_boru.head())
