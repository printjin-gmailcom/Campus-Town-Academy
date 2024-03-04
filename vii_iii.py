# -*- coding: utf-8 -*-
"""VII_III

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kXuf1SdUKPirGaHmDEjAwV18ryECok1A
"""

import seaborn as sns

raw=sns.load_dataset('tips')
raw

raw[['total_bill', 'tip']]

raw.info()

raw.head()

raw['tip'].hist()

raw['tip'].hist(bins=20)

sns.histplot(data = raw['tip'])

sns.relplot(data = raw, x = 'total_bill', y = 'tip', hue = 'smoker')

sns.relplot(data = raw, x = 'total_bill', y = 'tip', hue = 'day', kind='line')

sns.jointplot(data = raw, x = 'total_bill', y = 'tip', kind = 'kde')

sns.jointplot(data = raw, x = 'total_bill', y = 'tip', kind = 'hex')

sns.jointplot(data = raw, x = 'total_bill', y = 'tip', kind = 'reg')

sns.jointplot(data = raw, x = 'total_bill', y = 'tip', kind = 'hist')

sns.pairplot(data=raw)

sns.pairplot(data=raw, hue = 'day')

sns.boxplot(data = raw, x = 'day', y = 'tip')

raw.describe()

sns.swarmplot(data = raw, x = 'day', y = 'tip')

df = raw.pivot_table(index = 'day',
               columns = 'size',
               values = 'tip',
               aggfunc = 'mean')
df

sns.heatmap(data=df,
           cmap = 'Reds',
           annot = True, fmt = '.1%')
