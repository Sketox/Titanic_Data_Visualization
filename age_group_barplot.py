import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import cleaned dataframe with outliers removed
from titanic_visualization import df_no_outliers

# Load cleaned Titanic dataset
try:
    df = pd.read_csv("titanic_clean.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'titanic_clean.csv' not found.")
    exit()

# Style settings
plt.style.use("dark_background")
sns.set_palette("pastel")

# Main visualization script
def main():
    # Countplot: Passenger distribution by age group (unconditionally)
    sns.countplot(
        x="AgeGroup", data=df,
        order=["Minor", "Adult", "Senior"]
    )
    plt.title("Passenger Distribution by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Passenger Count")
    plt.savefig("images/age_group_distribution.png")
    plt.show()

    # Heatmap: Correlation matrix of selected numeric variables
    selected_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    if all(col in df_no_outliers.columns for col in selected_columns):
        correlation_matrix = df_no_outliers[selected_columns].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix, annot=True,
            cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Correlation Heatmap", fontsize=14)
        plt.savefig("images/correlation_heatmap.png")
        plt.show()
    else:
        print("⚠️ Warning: One or more selected columns are missing in df_no_outliers.")

# Script execution
if __name__ == "__main__":
    main()
