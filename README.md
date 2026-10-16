# Nobel Laureates Data Analysis

An exploratory data analysis project examining Nobel Prize laureates using Python, Pandas, NumPy, and Matplotlib.

## Project Overview

This project focuses on data preprocessing, feature engineering, statistical analysis, and data visualization using historical Nobel Prize laureate records.

Developed as part of a hands-on learning project on Hyperskill (JetBrains Academy).

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib

## Features

### Data Cleaning
- Identification of duplicate records
- Handling missing values
- Standardization of country names
- Extraction of birth countries

### Feature Engineering
- Extraction of birth years from different date formats
- Calculation of laureates' ages when receiving the Nobel Prize
- Grouping countries with fewer than 25 laureates

### Data Visualization
- Pie chart showing Nobel laureates by country of birth
- Grouped bar chart comparing male and female Nobel laureates
- Box plot showing the age distribution across Nobel Prize categories

## Project Structure

```text
nobel-laureates-data-analysis/
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/nobel-laureates-data-analysis.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset in a directory named `data`:

```text
data/
└── nobel_laureates.json
```

Run the application:

```bash
python main.py
```

## Dataset

The project uses a Nobel Prize laureates dataset provided through the Hyperskill learning platform.

The original dataset is not included in this repository.

## Educational Purpose

This repository documents practical experience in data cleaning, feature engineering, exploratory data analysis, and statistical visualization.
