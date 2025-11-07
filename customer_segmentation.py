import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

df = pd.read_csv('electricity_usage.csv')
df.head()

df_normalized = (df-df.min())/(df.max()-df.min())

K = 10
store_SSE = np.zeros(K)

for k in range(1, K+1):
  kmeanSpec = KMeans(n_clusters = k, n_init = 100)
  kmean_result = kmeanSpec.fit(df_normalized.loc[:, 't00':'t23'])
  store_SSE[k-1] = kmeanSpec.inertia_

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.rcParams['figure.figsize'] = [12,8]
plt.plot(range(1, K+1), store_SSE)
plt.xticks(range(1, K+1), fontsize = 18)
plt.yticks(fontsize = 18)
plt.ylabel("SSE",fontsize = 18)
plt.xlabel("number of clusters", fontsize = 18)

kmeanSpec = KMeans(n_clusters = 3, n_init = 100)
kmean_result = kmeanSpec.fit(df_normalized.loc[:, 't00':'t23'])
df["segment"] = kmean_result.labels_

hourly_columns = ['t' + str(i).zfill(2) for i in range(24)]

plt.figure(figsize=(10, 6))
for segment in df['segment'].unique():
    segment_data = df[df['segment'] == segment][hourly_columns].mean()

    plt.plot(np.arange(24), segment_data, label=f'Segment {segment}')

plt.xlabel('Hour of the Day')
plt.ylabel('Average Consumption')
plt.title('Electricity Consumption Patterns by Segment')
plt.xticks(np.arange(24), [f'{i}:00' for i in range(24)], rotation=90)
plt.legend(title='Segment')
plt.grid(True)
plt.tight_layout()
plt.show()

"""Segment 1 Mid-day users:  In this segment electricity consumption is moderate between 0:00 to 5:00. During 5:00 to 18:00 electricity consumption increases steadily and at 18:00 to 23:00, it starts declining.

Segment 0 Evening users: In this segment, electricity consumption has a steep decrease from 0:00 to 3:00. During 3:00 to 15:00, the electricity consumption is very low. At 15:00 to 23:00, there is a steep increase in the use the electricity, peaking at 22:00.

Segment 2 Morning users: In this segment electricity consumption is increasing from 0:00 to 7:00 and peaks at around 6:00 to 7:00. It starts decreasing from 7:00 to 13:00 and rises back up from 13:00 to 19:00. The energy consumption decreases fast from 19:00 to 23:00.
"""

print(df['segment'].count())
cluster_summary = df.groupby('segment')[['EmailEff', 'TextEff', 'MailEff']].mean()
print(cluster_summary)

summary_table = df.groupby("segment").aggregate({

  "segment": "count"
 }
 )
summary_table

"""Segment 0: Mail

Segment 1: Text

Segment 2: Email
"""