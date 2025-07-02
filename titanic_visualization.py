import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
try:
    df = pd.read_csv("titanic_clean.csv")
    print("✅ Dataset cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: El archivo CSV no se encontró.")
    exit()

# Configuración de estilo
plt.style.use("dark_background")
sns.set_palette("pastel") 

# Eliminar outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_no_outliers = remove_outliers_iqr(df, "Age")
df_no_outliers = remove_outliers_iqr(df_no_outliers, "Fare")

# 1. Gráfico de dispersión con línea de regresión
sns.lmplot(
    x="Age", y="Fare", data=df_no_outliers,
    height=6, aspect=1.5, line_kws={'color': 'cyan'}
)
plt.title("Relación entre Edad y Tarifa", fontsize=14)
plt.xlabel("Edad")
plt.ylabel("Tarifa")
plt.savefig("images/age_vs_fare.png")
plt.show()

# 2. Gráfico de violín
plt.figure(figsize=(8,6))
sns.violinplot(x="Pclass", y="Age", data=df_no_outliers, inner="box", linewidth=1.2)
plt.title("Distribución de Edad por Clase", fontsize=14)
plt.xlabel("Clase")
plt.ylabel("Edad")
plt.savefig("images/age_by_class.png")
plt.show()

# 3. Mapa de calor
columnas_utiles = ["Age", "Fare", "Pclass"]
correlaciones = df_no_outliers[columnas_utiles].corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlaciones, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor de Correlaciones", fontsize=14)
plt.savefig("images/correlation_heatmap.png")
plt.show()
