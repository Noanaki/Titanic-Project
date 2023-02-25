import pandas as pd
import numpy as np
import sklearn as sk

DataFrame = pd.read_csv('train.csv')
DataFrame = DataFrame.drop(['PassengerId', 'Ticket', 'Name'], axis=1)
