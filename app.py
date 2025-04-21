import io
import base64
from datetime import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

app = Flask(__name__, template_folder='.', static_folder='.')

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_png = None
    recommendation = None

    if request.method == 'POST':
        # 1. grab inputs
        ticker     = request.form['ticker'].upper().strip()
        start_date = request.form['start_date']
        end_date   = request.form['end_date']

        # 2. download and prep data
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})

        # 3. fit Prophet
        m = Prophet(yearly_seasonality='auto')
        m.fit(df)

        # 4. make future frame out to end_date
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)

        # 5. plot forecast
        fig = m.plot(forecast, xlabel='Date', ylabel='Price')
        add_changepoints_to_plot(fig.gca(), m, forecast)
        plt.title(f'{ticker} Price Forecast')

        # 6. grab image as PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_png = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        # 7. buy/sell recommendation
        current_price    = df['y'].iloc[-1]
        forecasted_price = forecast['yhat'].iloc[-1]
        if   forecasted_price > current_price: recommendation = "Buy"
        elif forecasted_price < current_price: recommendation = "Sell"
        else:                                   recommendation = "Hold"

    return render_template('index.html',
                           chart_png=chart_png,
                           recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
