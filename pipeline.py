"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = 'data/sales_records.csv'
OUTPUT_DIR = 'output'


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    """Load sales records from a CSV file."""
    
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} records from {filepath}")

    return df


def clean_data(df):
    """Handle missing values and fix data types."""

    df = df.copy()

    # Fill missing values
    df['quantity'] = df['quantity'].fillna(df['quantity'].median())
    df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows where both quantity and unit_price are missing
    df = df.dropna(subset=['quantity', 'unit_price'], how='all')

    print(f"Cleaned data: {len(df)} records")

    return df


def add_features(df):
    """Compute derived columns."""

    df = df.copy()

    df['revenue'] = df['quantity'] * df['unit_price']

    df['day_of_week'] = df['date'].dt.day_name()

    return df


def generate_summary(df):
    """Compute summary statistics."""

    total_revenue = df['revenue'].sum()

    avg_order_value = df['revenue'].mean()

    top_category = df.groupby('product_category')['revenue'].sum().idxmax()

    record_count = len(df)

    return {
        'total_revenue': total_revenue,
        'avg_order_value': avg_order_value,
        'top_category': top_category,
        'record_count': record_count
    }


def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Create and save 3 charts as PNG files."""

    os.makedirs(output_dir, exist_ok=True)

    # ─── Chart 1: Revenue by Product Category ──────────────────────────────
    revenue_by_category = df.groupby('product_category')['revenue'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(revenue_by_category.index, revenue_by_category.values)

    ax.set_title('Revenue by Product Category')
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Revenue')

    fig.savefig(f'{output_dir}/revenue_by_category.png', dpi=150, bbox_inches='tight')

    plt.close(fig)


    # ─── Chart 2: Daily Revenue Trend ─────────────────────────────────────
    daily_revenue = df.groupby('date')['revenue'].sum().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(daily_revenue.index, daily_revenue.values)

    ax.set_title('Daily Revenue Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')

    fig.savefig(f'{output_dir}/daily_revenue_trend.png', dpi=150, bbox_inches='tight')

    plt.close(fig)


    # ─── Chart 3: Avg Order Value by Payment Method ───────────────────────
    avg_payment = df.groupby('payment_method')['revenue'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(avg_payment.index, avg_payment.values)

    ax.set_title('Average Order Value by Payment Method')
    ax.set_xlabel('Average Order Value')
    ax.set_ylabel('Payment Method')

    fig.savefig(f'{output_dir}/avg_order_by_payment.png', dpi=150, bbox_inches='tight')

    plt.close(fig)


def main():
    """Run the full data pipeline end-to-end."""

    df = load_data(DATA_PATH)

    df = clean_data(df)

    df = add_features(df)

    summary = generate_summary(df)

    print("\n=== Summary ===")
    print(f"Total Revenue: {summary['total_revenue']}")
    print(f"Average Order Value: {summary['avg_order_value']}")
    print(f"Top Category: {summary['top_category']}")
    print(f"Record Count: {summary['record_count']}")

    create_visualizations(df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()