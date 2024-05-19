import yfinance as yf
import pandas as pd

# Fetch data for International Consolidated Airlines Group, S.A. (IAG.L) for the last 5 years
ticker = 'IAG.L'
iag = yf.Ticker(ticker)

# Get historical market data
historical_data = iag.history(period="5y")

# Function to safely fetch financial data
def fetch_financial_data(ticker):
    try:
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        return financials, balance_sheet, cash_flow
    except Exception as e:
        print(f"Error fetching financial data: {e}")
        return None, None, None

# Fetch financial data
financials, balance_sheet, cash_flow = fetch_financial_data(iag)

# Check if data is fetched successfully
if financials is not None and balance_sheet is not None:
    # Calculate financial ratios
    ratios = pd.DataFrame(index=financials.columns)

    # Net Profit Margin = Net Income / Total Revenue
    ratios.loc['Net Profit Margin'] = financials.loc['Net Income'] / financials.loc['Total Revenue']

    # Return on Assets (ROA) = Net Income / Total Assets
    ratios.loc['Return on Assets (ROA)'] = financials.loc['Net Income'] / balance_sheet.loc['Total Assets']

    # Return on Equity (ROE) = Net Income / Shareholders' Equity
    ratios.loc['Return on Equity (ROE)'] = financials.loc['Net Income'] / balance_sheet.loc["Total Stockholder Equity"]

    # Total Assets
    ratios.loc['Total Assets'] = balance_sheet.loc['Total Assets']

    # Total Revenue
    ratios.loc['Total Revenue'] = financials.loc['Total Revenue']

    # Display the calculated ratios
    print(ratios)
else:
    print("Failed to fetch financial data.")
