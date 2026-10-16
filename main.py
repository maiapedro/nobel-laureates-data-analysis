import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "data" / "nobel_laureates.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

COLORS = [
    "blue", "orange", "red", "yellow", "green",
    "pink", "brown", "cyan", "purple"
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place nobel_laureates.json inside the data directory."
        )

    df = pd.read_json(DATA_PATH)

    print("\n" + "=" * 60)
    print("NOBEL LAUREATES DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    return df


def extract_country(place):
    if not isinstance(place, str) or "," not in place:
        return None

    return place.split(",")[-1].strip()


def clean_data(df):
    df = df.copy()

    df.dropna(subset=["gender"], inplace=True)

    df["place_of_birth"] = df["place_of_birth"].apply(extract_country)

    df["born_in"] = df["born_in"].replace(
        r"^\s*$", np.nan, regex=True
    )

    df["born_in"] = df["born_in"].fillna(df["place_of_birth"])

    df.dropna(subset=["born_in"], inplace=True)

    df["born_in"] = df["born_in"].str.strip()

    country_map = {
        "US": "USA",
        "U.S.": "USA",
        "United States": "USA",
        "United Kingdom": "UK"
    }

    df["born_in"] = df["born_in"].replace(country_map)

    df["category"] = df["category"].replace(
        r"^\s*$", np.nan, regex=True
    )

    df.dropna(subset=["category"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print("\nCleaned dataset shape:")
    print(df.shape)

    return df


def extract_birth_year(date):
    if pd.isna(date):
        return np.nan

    date = str(date).strip()

    if len(date) == 4 and date.isdigit():
        return int(date)

    parsed_date = pd.to_datetime(
        date,
        format="mixed",
        errors="coerce"
    )

    if pd.isna(parsed_date):
        return np.nan

    return parsed_date.year


def feature_engineering(df):
    df = df.copy()

    df["year_born"] = df["date_of_birth"].apply(extract_birth_year)

    df["age_of_winning"] = df["year"] - df["year_born"]

    country_counts = df["born_in"].value_counts()

    df["born_in_corrected"] = df["born_in"].apply(
        lambda country: (
            country
            if country_counts[country] >= 25
            else "Other countries"
        )
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print("\nNew features:")
    print(
        df[
            ["name", "year_born", "age_of_winning", "born_in_corrected"]
        ].head(10)
    )

    return df


def analyze_data(df):
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nTotal Nobel laureates:")
    print(len(df))

    print("\nLaureates by category:")
    print(df["category"].value_counts())

    print("\nLaureates by gender:")
    print(df["gender"].value_counts())

    print("\nTop 10 countries of birth:")
    print(df["born_in"].value_counts().head(10))

    print("\nAge statistics:")
    print(df["age_of_winning"].describe())

    print("\nAverage age by category:")
    print(
        df.groupby("category")["age_of_winning"]
        .mean()
        .sort_values()
    )


def plot_countries(df):
    counts = df["born_in_corrected"].value_counts()

    labels = counts.index
    sizes = counts.values

    explode = [
        0.08 if i >= 3 else 0
        for i in range(len(labels))
    ]

    def autopct_format(pct):
        absolute = round(pct * sizes.sum() / 100)
        return f"{pct:.2f}%\n({absolute:.0f})"

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.pie(
        sizes,
        labels=labels,
        colors=COLORS[:len(labels)],
        explode=explode,
        autopct=autopct_format,
        startangle=0,
        textprops={"fontsize": 11}
    )

    ax.set_title(
        "Nobel Laureates by Country of Birth",
        fontsize=20
    )

    fig.savefig(
        OUTPUT_DIR / "nobel_laureates_by_country.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_gender_distribution(df):
    gender_counts = pd.crosstab(
        df["category"],
        df["gender"]
    )

    gender_counts = gender_counts.reindex(
        columns=["male", "female"],
        fill_value=0
    )

    categories = gender_counts.index.to_list()

    males = gender_counts["male"].to_numpy()
    females = gender_counts["female"].to_numpy()

    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.bar(
        x - 0.2,
        males,
        width=0.4,
        color="blue",
        label="Males"
    )

    ax.bar(
        x + 0.2,
        females,
        width=0.4,
        color="crimson",
        label="Females"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories,
        rotation=45,
        ha="right",
        fontsize=14
    )

    ax.tick_params(axis="y", labelsize=14)

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Nobel Laureates Count", fontsize=14)

    ax.set_title(
        "Male and Female Nobel Prize Winners by Category",
        fontsize=20
    )

    ax.legend(fontsize=14)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "nobel_laureates_by_gender.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_age_distribution(df):
    categories = [
        "Chemistry",
        "Economics",
        "Literature",
        "Peace",
        "Physics",
        "Physiology or Medicine",
        "All categories"
    ]

    valid_df = df.dropna(subset=["age_of_winning"])

    ages = [
        valid_df.loc[
            valid_df["category"] == category,
            "age_of_winning"
        ].to_numpy()
        for category in categories[:-1]
    ]

    ages.append(valid_df["age_of_winning"].to_numpy())

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.boxplot(
        ages,
        tick_labels=categories,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "green",
            "markeredgecolor": "green"
        }
    )

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Age of Winning", fontsize=14)

    ax.set_title(
        "Age Distribution of Nobel Prize Winners by Category",
        fontsize=20
    )

    ax.tick_params(axis="both", labelsize=14)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "nobel_laureates_age_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def generate_visualizations(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_countries(df)
    print("Country distribution chart generated.")

    plot_gender_distribution(df)
    print("Gender distribution chart generated.")

    plot_age_distribution(df)
    print("Age distribution box plot generated.")

    print(f"\nCharts saved in: {OUTPUT_DIR}")


def main():
    df = load_data()

    df = clean_data(df)

    df = feature_engineering(df)

    analyze_data(df)

    generate_visualizations(df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
