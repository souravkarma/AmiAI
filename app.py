import io
import base64

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='.', static_folder='.')

def pred(ticker, start_date, end_date):
    # 1. Download and reset index
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df.reset_index()

    # 2. Build the ds/y DataFrame
    df2 = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    # ensure y is numeric
    df2['y'] = pd.to_numeric(df2['y'], errors='coerce')
    df2 = df2.dropna(subset=['y'])

    # 3. Fit Prophet on the clean data
    m = Prophet(yearly_seasonality='auto')
    m.fit(df2)

    # 4. Forecast 90 days out
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

    # 5. Plot and capture as PNG
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.title(f"{ticker} Price Forecast")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart_png = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    # 6. Recommendation logic
    current_price    = df2['y'].iloc[-1]
    forecasted_price = forecast['yhat'].iloc[-1]
    if   forecasted_price > current_price: recommendation = "Buy"
    elif forecasted_price < current_price: recommendation = "Sell"
    else:                                   recommendation = "Hold"

    return chart_png, recommendation

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_png      = None
    recommendation = None

    if request.method == 'POST':
        ticker     = request.form['ticker'].upper().strip()
        start_date = request.form['start_date']
        end_date   = request.form['end_date']
        chart_png, recommendation = pred(ticker, start_date, end_date)

    return render_template('index.html',
                           chart_png=chart_png,
                           recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
