import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import yfinance as yf

app = Flask(__name__, template_folder='.', static_folder='.')

def pred(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)

    # Check if we actually got data
    if df.empty:
        return None, "No data found for this ticker and date range."

    # Reset index and format for Prophet
    df = df.reset_index()
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Force 'y' to numeric and drop rows with NaN
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()

    # Ensure we still have data after cleaning
    if df.empty:
        return None, "Cleaned data is empty. Cannot forecast."

    # Forecast
    m = Prophet(yearly_seasonality='auto')
    m.fit(df)

    # Forecast 90 days into the future
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

    # Plot forecast
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.title(f"{ticker} Price Forecast")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    chart_png = base64.b64encode(buf.read()).decode()
    plt.close()

    # Buy/sell logic
    current_price = df['y'].iloc[-1]
    forecasted_price = forecast['yhat'].iloc[-1]
    if forecasted_price > current_price:
        recommendation = "Buy"
    elif forecasted_price < current_price:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return chart_png, recommendation

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_png = None
    recommendation = None
    error = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper().strip()
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        try:
            chart_png, recommendation = pred(ticker, start_date, end_date)
            if chart_png is None:
                error = recommendation
                recommendation = None
        except Exception as e:
            error = f"An error occurred: {e}"

    return render_template('index.html', chart_png=chart_png, recommendation=recommendation, error=error)

if __name__ == '__main__':
    app.run(debug=True)
