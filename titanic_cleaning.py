# -----------------------------------------------
# 🛳️ Titanic Dataset - Limpieza y Análisis Exploratorio
# -----------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------
# 1. Cargar datos
# -----------------------------------------------
try:
    df = pd.read_csv("titanic_raw.csv")
    print("✅ Dataset cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: El archivo CSV no se encontró.")
    exit()

# -----------------------------------------------
# 2. Revisión de valores nulos
# -----------------------------------------------
print("\n🔍 Valores nulos en el DataFrame:")
print(df.isnull().sum())

# -----------------------------------------------
# 3. Rellenar valores nulos
# -----------------------------------------------

# Rellenar Age según Pclass y Sex
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.mean())
)

# Rellenar Fare según Pclass
df["Fare"] = df.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.mean())
)

# Validar y reemplazar valores inválidos en Embarked
valid_embarked = ["C", "S", "Q"]
invalid_ports = ~df["Embarked"].isin(valid_embarked)
mask_invalid_or_missing = invalid_ports | df["Embarked"].isnull()

df.loc[mask_invalid_or_missing, "Embarked"] = df.loc[mask_invalid_or_missing].apply(
    lambda row: df[df["Pclass"] == row["Pclass"]]["Embarked"].mode().iloc[0],
    axis=1
)

# -----------------------------------------------
# 4. Eliminar duplicados
# -----------------------------------------------

# Eliminar espacios en blanco en cadenas (string cleaning)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Detectar duplicados ignorando PassengerId
columns_to_check = df.columns.drop("PassengerId")
duplicates = df[df.duplicated(subset=columns_to_check)]
print(f"\n📄 Duplicados detectados: {len(duplicates)}")
if not duplicates.empty:
    print(duplicates)

# Eliminar duplicados
df = df.drop_duplicates(subset=columns_to_check)

# -----------------------------------------------
# 5. Verificar y corregir inconsistencias en 'Sex' según título
# -----------------------------------------------

# Extraer título desde Name (ej: "Mr", "Mrs", etc.)
df["Title"] = df["Name"].str.extract(r",\s*(Mr|Mrs|Miss|Master|Ms|Dr)\.?\s")


# Detectar conflictos entre título y sexo
inconsistent_sex = df[
    ((df["Title"] == "Mr") & (df["Sex"] == "female")) |
    ((df["Title"].isin(["Mrs", "Miss", "Ms"])) & (df["Sex"] == "male"))
]

print(f"\n⚠️ Inconsistencias sexo/título encontradas: {len(inconsistent_sex)}")
if not inconsistent_sex.empty:
    print(inconsistent_sex[["Name", "Sex", "Title"]])

# Corregir valores incorrectos en 'Sex' basados en 'Title'
df.loc[df["Title"] == "Mr", "Sex"] = "male"
df.loc[df["Title"].isin(["Mrs", "Miss", "Ms"]), "Sex"] = "female"

# -----------------------------------------------
# 6. Detectar y eliminar outliers (Age y Fare) usando IQR
# -----------------------------------------------

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_no_outliers_age = remove_outliers_iqr(df, "Age")
df_no_outliers_fare = remove_outliers_iqr(df, "Fare")



def clasificar_edad(edad):
    if edad < 18:
        return "Minor"
    elif edad < 60:
        return "Adult"
    else:
        return "Senior"

df["AgeGroup"] = df["Age"].apply(clasificar_edad)


# -----------------------------------------------
# 7. Análisis: Relación entre Pclass y Supervivencia
# -----------------------------------------------

# Calcular tasa de supervivencia por clase
print("\n📊 Porcentaje de sobrevivientes por clase:")
survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100
print(survival_by_class.round(2))

# Tabla cruzada Pclass vs Survived
cross_tab = pd.crosstab(df["Pclass"], df["Survived"], margins=True)
print("\n📋 Tabla cruzada de supervivencia por clase:")
print(cross_tab)

# -----------------------------------------------
# ✅ Fin del proceso
# -----------------------------------------------
print("\n✅ Limpieza y análisis completados con éxito.")

# Guardar el DataFrame limpio en un nuevo archivo CSV
df.to_csv("titanic_clean.csv", index=False)
print("✅ Dataset limpio guardado como 'titanic_clean.csv'")

