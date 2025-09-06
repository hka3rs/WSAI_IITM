import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

'''
sns.countplot(x='x', data=df)
plt.show()
'''
df = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})


trips = pd.read_excel(r'tripDetails.xlsx')
print(trips.head())

sns.boxplot(x='TripLength', data=trips)
plt.show()