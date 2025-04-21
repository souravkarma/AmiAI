import io
import base64

from flask import Flask, render_template, request
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import yfinance as yf
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='.', static_folder='.')

def pred(x, start_date, end_date):
    # Download stock data
    df = yf.download(x, start=start_date, end=end_date)
    # flatten multi‑level columns if any
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()

    # Function to format dataframe for forecasting
    def df_formatting(df_in):
        df2 = df_in.loc[:, ['Date', 'Close']]
        df2.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        return df2

    # Function for price forecasting
    def price_forecasting(df_in, period):
        m = Prophet(yearly_seasonality='auto')
        m.fit(df_in)
        future = m.make_future_dataframe(periods=period)
        forecasts = m.predict(future)

        # Plot the forecasts
        fig = m.plot(forecasts)
        add_changepoints_to_plot(fig.gca(), m, forecasts)
        plt.title(x)

        # Also plot components (optional)
        m.plot_components(forecasts)

        return forecasts, fig

    # Function to provide buy/sell/hold recommendation
    def buy_sell_recommendation(current_price, forecasted_price):
        if forecasted_price > current_price:
            return "Buy"
        elif forecasted_price < current_price:
            return "Sell"
        else:
            return "Hold"

    # 1. Prepare data
    df_prepped = df_formatting(df)
    # 2. Forecast out 90 days (same as your snippet)
    forecast, fig = price_forecasting(df_prepped, 90)

    # 3. Determine recommendation
    current_price    = df['Close'].iloc[-1]
    forecasted_price = forecast['yhat'].iloc[-1]
    recommendation   = buy_sell_recommendation(current_price, forecasted_price)

    # 4. Convert matplotlib fig to base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close('all')

    return png_b64, recommendation

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
