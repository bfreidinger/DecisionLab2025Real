import yfinance as yf
import requests

# Function to get the risk-free rate (US 10-Year Treasury yield)
def get_risk_free_rate():
    try:
        response = requests.get("https://www.marketwatch.com/investing/bond/tmubmusd10y")
        if response.status_code == 200:
            # Extracting the rate (This might require more parsing depending on MarketWatch changes)
            import re
            match = re.search(r'(\d+\.\d+)%', response.text)
            if match:
                return float(match.group(1)) / 100  # Convert percentage to decimal
    except:
        pass
    return 0.04  # Default to 4% if request fails

# Function to calculate expected return for multiple stocks using CAPM
def expected_returns(tickers, market_return=0.08):
    rf = get_risk_free_rate()  # Get risk-free rate once
    results = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        beta = stock.info.get('beta', None)

        if beta is None:
            print(f"Warning: Could not retrieve beta for {ticker}")
            continue  # Skip this stock if beta is missing

        # Apply CAPM formula: ra = rf + beta * (rm - rf)
        ra = rf + beta * (market_return - rf)
        
        results[ticker] = ra  # Store expected return in dictionary

    return results

# Example usage
#tickers_list = ["AAPL", "MSFT", "AMZN", "TSLA"]  # Add any ticker symbols
#results = expected_returns(tickers_list)

# Print results
#print("\nExpected Returns:")
#for ticker, exp_return in results.items():
#    print(f"{ticker}: {exp_return:.4f}")
