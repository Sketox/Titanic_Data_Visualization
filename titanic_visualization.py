import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 🔧 Helper function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

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

# Remove outliers for age and fare
df_no_outliers = remove_outliers_iqr(df, "Age")
df_no_outliers = remove_outliers_iqr(df_no_outliers, "Fare")
df_no_outliers_age = remove_outliers_iqr(df, "Age")
df_no_outliers_fare = remove_outliers_iqr(df, "Fare")

# Main function for data visualization
def main():

    # Boxplot: Age before and after removing outliers
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df["Age"])
    plt.title("Age - Before Removing Outliers")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_no_outliers_age["Age"])
    plt.title("Age - After Removing Outliers")
    plt.tight_layout()
    plt.savefig("images/age_before_after.png")
    plt.show()

    # Boxplot: Fare before and after removing outliers
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df["Fare"])
    plt.title("Fare - Before Removing Outliers")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_no_outliers_fare["Fare"])
    plt.title("Fare - After Removing Outliers")
    plt.tight_layout()
    plt.savefig("images/fare_before_after.png")
    plt.show()

    # Countplot: Passenger class by embarkation port
    sns.countplot(x="Embarked", hue="Pclass", data=df)
    plt.title("Passenger Class by Embarkation Port")
    plt.savefig("images/class_by_port.png")
    plt.show()

    # Barplot: Survival rate by class
    sns.barplot(x="Pclass", y="Survived", data=df, ci=None)
    plt.title("Survival Rate by Class")
    plt.xlabel("Passenger Class (Pclass)")
    plt.ylabel("Survival Rate")
    plt.ylim(0, 1)
    plt.savefig("images/survival_by_class.png")
    plt.show()

    # Countplot: Survival by age group
    if "AgeGroup" in df.columns:
        sns.countplot(
            x="AgeGroup", hue="Survived", data=df,
            order=["Minor", "Adult", "Senior"]
        )
        plt.title("Survival by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Passenger Count")
        plt.legend(title="Survived", labels=["No", "Yes"])
        plt.savefig("images/survival_by_age_group.png")
        plt.show()

    # Scatterplot with regression line: Age vs Fare
    sns.lmplot(
        x="Age", y="Fare", data=df_no_outliers,
        height=6, aspect=1.5, line_kws={'color': 'cyan'}
    )
    plt.title("Age vs Fare", fontsize=14)
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.savefig("images/age_vs_fare.png")
    plt.show()

    # Violin plot: Age distribution by class
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x="Pclass", y="Age", data=df_no_outliers,
        inner="box", linewidth=1.2
    )
    plt.title("Age Distribution by Class", fontsize=14)
    plt.xlabel("Passenger Class")
    plt.ylabel("Age")
    plt.savefig("images/age_by_class.png")
    plt.show()

# 🏁 Script execution

if __name__ == "__main__":
    main()
