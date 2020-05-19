import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

file_path = os.path.join("cleanedbigfoodlist(numberonly).csv")
dirty_df = pd.read_csv(file_path)
indexed_df = dirty_df.set_index("NDB_No")
# print(indexed_df.head())
num_df = indexed_df.drop(columns=["Shrt_Desc", "Weight_desc(Unit)", "GmWt_Desc2(Unit)"])
# num_df = num_df.astype(float)
# print(num_df.head())


# Define N_neighbors value
imputer = KNNImputer(n_neighbors=5)

# Impute/Fill Missing Values
df_filled = imputer.fit_transform(num_df)

# print(df_filled)
# print(type(df_filled))

columns = columns = ["Water", "Energy", "Protein", "Lipid_Total", "Carbohydrate", "Fiber", "Sugar_Total", "Calcium", "Iron", "Magnesium", 
            "Phosphorus", "Potassium", "Sodium", "Zinc", "Copper", "Manganese", "Selenium", "Vitamin_C", "Thiamin", "Riboflavin", 
            "Niacin", "Panto_Acid", "Vitamin_B6", "Folate_Total", "Folic_Acid", "Food_Folate_mcg", "Folate_DFE_mcg", "Choline_Tot_mg", "Vitamin_B12", "Vit_A_IU",
            "Vitamin_A", "Retinol", "Alpha_Carot_mcg", "Beta_Carot_mcg", "Beta_Crypt_mcg", "Lycopene_mcg", "Lut+Zea_mcg", "Vitamin_E", "Vitamin_D", "Vit_D_IU", 
            "Vitamin_K", "FA_Sat_g", "FA_Mono_g", "FA_Poly_g", "Cholestrol", "Weight_grams", "Weight_desc", "GmWt_2", "GmWt_Desc2", "Refuse_Pct"]

df = pd.DataFrame(data=df_filled, columns=columns)

# print(df.head())

df.to_csv("cleanedbigfooddata.csv")
