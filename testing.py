import pandas as pd
from dataframe_to_image import dataframe_to_image
import dataframe_image as dfi

df = pd.DataFrame(columns=['xasdasd', 'yasd', 'zaasdasdasdasdasdassd', 'a'])

print(df.head(5))

df.loc[len(df.index)] = ['Amy', 89, 144, 3360]
df.loc[len(df.index)] = ['Amya', 899, 996, 15]
df.loc[len(df.index)] = ['Amyak', 899, 996, 105]
df.loc[len(df.index)] = ['Asmya', 1899, 96, 750]

print(df.head(5))
df = df.set_index('xasdasd')

df_style = df.apply(pd.to_numeric).style.background_gradient(subset=['a', 'yasd'])

df =df.sort_values(by=['yasd', 'a'], ascending=False)
print(df.head(5))



#print(df_styled)
pd.set_option('display.max_columns', None)

dfi.export(df, "df.png", dpi = 2048, table_conversion="selenium", max_rows=-1)
dfi.export(df_style, "df_style.png", dpi = 1024)

print(df_style)


#dataframe_to_image.convert(df_styled,visualisation_library='matplotlib')
