from ydata_profiling import ProfileReport

import pandas as pd
df=pd.read_csv(r"C:\Users\zaizzhaim\OneDrive - Mubea\Desktop\02112\Book2.csv")
df=df.dropna()
df=df[df['AP4_Check_AVG']>35]

profile = ProfileReport(df, title="Ydata Profiling Report")
profile.to_file(r"C:\Users\zaizzhaim\OneDrive - Mubea\Desktop\02112\ydata_EDA.html")
df.describe(percentiles=[0.001,0.005,0.01,0.9,0.95,0.99,0.995,0.999])
from matplotlib import pyplot as plt

plt.scatter(range(len(df)),df,color='r',s=10)
plt.axhline(38,color='black',linestyle="dashed")
plt.axhline(39.3,color='black',linestyle="dashed")
plt.show()

df.describe(percentiles=[0.001,0.005,0.01,0.9,0.95,0.99,0.995,0.999])
for i in [[38.3,39.3],[38,40],[37,40]]:
    print(f'Limits:{i}')
    df_2 = df[(df['AP4_Check_AVG'] > i[0]) & (df['AP4_Check_AVG'] < i[1])]
    print(f'NOK:{len(df)-len(df_2)}')
    plt.scatter(range(len(df)),df,color='r',s=10)
    plt.axhline(i[0],color='black',linestyle="dashed")
    plt.axhline(i[1],color='black',linestyle="dashed")
    plt.show()
    sns.histplot(df, kde=True)
    plt.title("Histogram + KDE")
    plt.axvline(i[0],color='black',linestyle="dashed")
    plt.axvline(i[1],color='black',linestyle="dashed")
    plt.show()
    