import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os # Import the os module here
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Danh sách các mã cổ phiếu mẫu
symbols = ['AAPL', 'NVDA', 'SONY', 'INTC', 'MSFT']

# Create the 'dataset' directory if it doesn't exist
import os
if not os.path.exists('dataset'):
    os.makedirs('dataset')

def safe_float(x):
    """Safely convert Pandas Series or single values to float"""
    try:
        if isinstance(x, pd.Series):  # Kiểm tra nếu đầu vào là Pandas Series
            return float(x.iloc[0])  # Lấy giá trị đầu tiên và chuyển sang float
        return float(x)  # Chuyển trực tiếp sang float nếu là giá trị đơn
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert {x} to float. Error: {e}")
        return None  # Trả về None nếu lỗi

def get_stock_data(symbol, start_date, end_date=datetime.now()):
    """
    Lấy dữ liệu lịch sử cổ phiếu từ Yahoo Finance.
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None
def preprocess_stock_data(df):
    """
    Tiền xử lý dữ liệu cổ phiếu từ file CSV
    """
    try:
        # 1. Xử lý cột Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # 2. Đảm bảo có đầy đủ các cột cần thiết
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 3. Chuyển đổi kiểu dữ liệu
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # 4. Xử lý giá trị null và bất thường
        df = df.dropna()
        df = df[df['Volume'] > 0]  # Loại bỏ các ngày không có giao dịch

        # 5. Sắp xếp theo thời gian
        df = df.sort_index()

        # 6. Thêm các chỉ báo kỹ thuật cơ bản
        df['MA20'] = df['Adj Close'].rolling(window=20).mean()
        df['MA50'] = df['Adj Close'].rolling(window=50).mean()

        return df

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý dữ liệu: {str(e)}")
        return None

def clean_data_with_header(df, symbol):
    """
    Xử lý dữ liệu và thêm hàng tiêu đề chứa mã cổ phiếu.
    """
    # Bước 1: Loại bỏ hàng tiêu đề thừa
    df = df.iloc[1:].reset_index(drop=True)

    # Bước 2: Đặt tên cho cột đầu tiên là 'Date'
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Bước 3: Chuyển cột 'Date' sang định dạng datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Bước 4: Xóa các hàng thiếu dữ liệu
    df.dropna(inplace=True)

    # Bước 5: Thêm hàng tiêu đề chứa mã cổ phiếu
    new_header = pd.DataFrame([[symbol] + [""] * (df.shape[1] - 1)], columns=df.columns)
    df = pd.concat([new_header, df], ignore_index=True)

    # Bước 6: Reset lại chỉ mục
    df.reset_index(drop=True, inplace=True)

    return df


def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # RSI Đánh giá tình trạng quá mua/quá bán.
    delta = df['Close'].diff() # Tính chênh lệch giữa giá đóng cửa ngày hiện tại và trước
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()# Lấy trung bình tăng
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()# Lấy trung bình giảm
    rs = gain / loss # Tính tỷ lệ tăng/giảm
    df['RSI'] = 100 - (100 / (1 + rs)) # Công thức RSI

    # MACD Xác định xu hướng giá.
    exp1 = df['Close'].ewm(span=12, adjust=False).mean() # EMA 12 ngày
    exp2 = df['Close'].ewm(span=26, adjust=False).mean() # EMA 26 ngày
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()# Đường tín hiệu

    # Bollinger Bands Xác định biến động giá.
    rolling_mean = df['Close'].rolling(window=20).mean()# Trung bình 20 ngày
    rolling_std = df['Close'].rolling(window=20).std() # Độ lệch chuẩn 20 ngày

    df['BB_middle'] = rolling_mean
    df['BB_upper'] = rolling_mean + (2 * rolling_std)  # Sửa lại
    df['BB_lower'] = rolling_mean - (2 * rolling_std)  # Sửa lại

    # Volume MA Trung bình khối lượng giao dịch
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    return df

def predict_prices(df, days):
    """Predict future prices using enhanced algorithm"""
    if df is None or len(df) < 20: # Kiểm tra dữ liệu đủ dài
        return None

    closes = df['Close'].values.flatten() # Lấy giá đóng cửa thành mảng
    ma20 = pd.Series(closes).rolling(20).mean() # Trung bình động 20 ngày
    std = safe_float(closes[-20:].std())  # Độ lệch chuẩn

    last_price = safe_float(closes[-1]) # Giá cuối cùng
    trend = safe_float(ma20.iloc[-1] - ma20.iloc[-20]) / 20 if len(ma20) >= 20 else 0 # Xu hướng

    # Enhanced prediction with technical indicators
    rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
    macd = df['MACD'].iloc[-1] if 'MACD' in df else 0

    # Adjust trend based on technical indicators
    if rsi > 70:
        trend *= 0.8  # Reduce upward trend if overbought
    elif rsi < 30:
        trend *= 1.2  # Increase upward trend if oversold

    if macd > 0:
        trend *= 1.1  # Increase trend if MACD is positive
    else:
        trend *= 0.9  # Decrease trend if MACD is negative

    predictions = []
    current_price = last_price

    for _ in range(days):
        # Add more sophisticated random variation (Độ biến động)
        volatility = std * 0.1
        technical_factor = (rsi - 50) / 500  # Small adjustment based on RSI
        random_change = np.random.normal(0, volatility)

        current_price += trend + random_change + technical_factor
        predictions.append(max(0, current_price))  # Ensure price doesn't go negative

    return predictions

def calculate_metrics(df, predictions, forecast_days):
    """Calculate enhanced metrics"""
    last_price = safe_float(df['Close'].iloc[-1])
    pred_price = safe_float(predictions[0])
    avg_price = float(sum(predictions) / len(predictions))
    change = ((pred_price - last_price) / last_price) * 100

    # Calculate additional metrics
    historical_volatility = safe_float(df['Close'].pct_change().std() * np.sqrt(252) * 100)
    max_prediction = max(predictions)
    min_prediction = min(predictions)
    pred_volatility = np.std(predictions) / np.mean(predictions) * 100

    # Add technical metrics
    rsi = safe_float(df['RSI'].iloc[-1]) if 'RSI' in df else None
    macd = safe_float(df['MACD'].iloc[-1]) if 'MACD' in df else None
    signal = safe_float(df['Signal'].iloc[-1]) if 'Signal' in df else None

    # Calculate trend strength Tính độ mạnh của xu hướng (Trend Strength) Nếu MA20 cao hơn MA50, xu hướng ngắn hạn tăng mạnh. Ngược lại, nếu MA20 thấp hơn MA50, xu hướng giảm.
    ma20 = df['Close'].rolling(window=20).mean()
    ma50 = df['Close'].rolling(window=50).mean()
    # Chuyển đổi sang float
    trend_strength = safe_float(((ma20.iloc[-1] / ma50.iloc[-1]) - 1) * 100)

    return {
        'last_price': last_price,
        'pred_price': pred_price,
        'avg_price': avg_price,
        'change': change,
        'historical_volatility': historical_volatility,
        'max_prediction': max_prediction,
        'min_prediction': min_prediction,
        'pred_volatility': pred_volatility,
        'rsi': rsi,
        'macd': macd,
        'signal': signal,
        'trend_strength': trend_strength
    }
##Tạo biểu đồ hiển thị dữ liệu cổ phiếu với các chỉ báo kỹ thuật.


def create_macd_chart(df, symbol):
    """Create separate MACD chart"""
    fig = go.Figure()

    # MACD Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ))

    # Signal Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        name='Signal',
        line=dict(color='orange')
    ))

    # MACD Histogram
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD'] - df['Signal'],
        name='MACD Histogram',
        marker_color='gray'
    ))

    fig.update_layout(
        title=f'MACD - {symbol}',
        height=300,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig

def create_rsi_chart(df, symbol):
    """Create separate RSI chart"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ))

    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(
        title=f'RSI - {symbol}',
        height=300,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[0, 100])
    )

    return fig
def display_enhanced_metrics(metrics):
    """Display enhanced metrics with tooltips"""
    st.subheader("Detailed Metrics")

    # Define tooltips
    tooltips = {
        'price': "Latest closing price of the stock",
        'volatility': "Historical volatility based on last 252 trading days",
        'prediction': "Predicted price for next trading day",
        'rsi': "Relative Strength Index (Oversold < 30, Overbought > 70)",
        'macd': "Moving Average Convergence Divergence",
        'trend': "Trend strength based on MA20 vs MA50"
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.metric("Current Price",
                     f"${metrics['last_price']:.2f}")
            st.markdown(f"<div class='tooltip'>ℹ️ {tooltips['price']}</div>",
                       unsafe_allow_html=True)

        with st.container():
            st.metric("Historical Volatility",
                     f"{metrics['historical_volatility']:.1f}%")
            st.markdown(f"<div class='tooltip'>ℹ️ {tooltips['volatility']}</div>",
                       unsafe_allow_html=True)

    with col2:
        with st.container():
            st.metric("Tomorrow's Prediction",
                     f"${metrics['pred_price']:.2f}",
                     f"{metrics['change']:+.2f}%")
            st.markdown(f"<div class='tooltip'>ℹ️ {tooltips['prediction']}</div>",
                       unsafe_allow_html=True)

        with st.container():
            if metrics['rsi'] is not None:
                st.metric("RSI",
                         f"{metrics['rsi']:.1f}")
                st.markdown(f"<div class='tooltip'>ℹ️ {tooltips['rsi']}</div>",
                          unsafe_allow_html=True)

    with col3:
        with st.container():
            if metrics['macd'] is not None:
                st.metric("MACD",
                         f"{metrics['macd']:.2f}")
                st.markdown(f"<div class='tooltip'>ℹ️ {tooltips['macd']}</div>",
                          unsafe_allow_html=True)

        with st.container():
            st.metric("Trend Strength",
                     f"{metrics['trend_strength']:+.2f}%")
            st.markdown(f"<div class='tooltip'>ℹ️ {tooltips['trend']}</div>",
                       unsafe_allow_html=True)

    # Add custom CSS for tooltips
    st.markdown("""
    <style>
    .tooltip {
        font-size: 0.8em;
        color: gray;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def add_settings_sidebar():
    """Add settings sidebar"""
    with st.sidebar:
        st.subheader("Display Settings")

        theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            help="Change between light and dark theme"
        )

        indicators = st.multiselect(
            "Technical Indicators",
            ["Bollinger Bands", "MACD", "RSI", "MA20", "MA50"],
            default=["MA20", "MA50","Bollinger Bands", "MACD", "RSI"],
            help="Select technical indicators to display"
        )

        chart_height = st.slider(
            "Chart Height",
            min_value=400,
            max_value=1000,
            value=800,
            step=50,
            help="Adjust chart height"
        )

        st.subheader("Analysis Settings")

        prediction_confidence = st.slider(
            "Prediction Confidence",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust prediction confidence interval"
        )

        return {
            "theme": theme,
            "indicators": indicators,
            "chart_height": chart_height,
            "prediction_confidence": prediction_confidence
        }

def display_prediction_table(future_dates, predictions, metrics):
    """Display prediction table with enhanced formatting"""
    st.subheader("Detailed Predictions")

    df_pred = pd.DataFrame({
        'Date': future_dates,  # Already named "Date"
        'Predicted Price': [f"${p:.2f}" for p in predictions],
        'Change (%)': [
            f"{((p - metrics['last_price']) / metrics['last_price'] * 100):+.2f}%"
            for p in predictions
        ],
        'Confidence Interval': [
            f"${p-p*0.05:.2f} - ${p+p*0.05:.2f}"
            for p in predictions
        ]
    })

    # Add styling
    def highlight_changes(val):
        if '%' in str(val):
            num = float(val.strip('%').replace('+', ''))
            if num > 0:
                return 'color: green'
            elif num < 0:
                return 'color: red'
        return ''

    styled_df = df_pred.style.applymap(highlight_changes)
    st.dataframe(styled_df, height=400)

def calculate_statistics(df):
    """Tính toán các tham số thống kê cho DataFrame."""
    # Loại bỏ cột 'Date' khỏi thống kê
    df_numeric = df.select_dtypes(include=np.number).drop(columns=['Date'], errors='ignore')

    statistics = df_numeric.describe().to_dict()  # Tính toán các tham số cơ bản
    for col in df_numeric.columns:
        # Tính toán các tham số bổ sung
        statistics[col]['Mode'] = df_numeric[col].mode()[0]
        statistics[col]['Sample Variance'] = df_numeric[col].var()
        statistics[col]['Kurtosis'] = df_numeric[col].kurt()
        statistics[col]['Skewness'] = df_numeric[col].skew()
        statistics[col]['Range'] = df_numeric[col].max() - df_numeric[col].min()
        statistics[col]['Sum'] = df_numeric[col].sum()
        statistics[col]['Count'] = df_numeric[col].count()
        # Confidence Level (95.0%)
        confidence_interval = stats.t.interval( # Sử dụng stats thay vì st
            0.95, len(df_numeric[col]) - 1, loc=np.mean(df_numeric[col]), scale=stats.sem(df_numeric[col])
        )
        statistics[col]['Confidence Level(95.0%)'] = f"{confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}"
    stats_df = pd.DataFrame(statistics) # Tạo DataFrame từ statistics

    return stats_df # Trả về DataFrame


def create_chart(df, start_date, end_date):
    """Create combined line chart for Adj Close, Close, Open with volume bar chart."""

    # Chuyển đổi start_date và end_date thành kiểu datetime64[ns]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Chuyển đổi cột 'Date' trong df thành kiểu datetime64[ns] nếu cần
    df['Date'] = df['Date'].dt.tz_localize(None)

    # Lọc DataFrame theo start_date và end_date
    mask = (df['Date'] >= start_date.to_numpy()) & (df['Date'] <= end_date.to_numpy())
    filtered_df = df.loc[mask]

    # Tính toán tổng hàng ngày cho mỗi biến
    filtered_df['Daily Adj Close Sum'] = filtered_df['Adj Close']
    filtered_df['Daily Close Sum'] = filtered_df['Close']
    filtered_df['Daily Open Sum'] = filtered_df['Open']
    filtered_df['Daily Volume Sum'] = filtered_df['Volume'] # Thêm cột tổng Volume theo ngày

    # Tạo biểu đồ đường với trục y phụ cho Volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Thêm các đường cho Adj Close, Close, Open
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Daily Adj Close Sum'], mode='lines', name='Daily Adj Close Sum'), secondary_y=False)
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Daily Close Sum'], mode='lines', name='Daily Close Sum'), secondary_y=False)
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Daily Open Sum'], mode='lines', name='Daily Open Sum'), secondary_y=False)

    # Thêm cột cho Volume sum theo ngày (trục y phụ)
    fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Daily Volume Sum'], name='Volume'), secondary_y=True) # Sử dụng Daily Volume Sum

    # Cập nhật layout biểu đồ đường
    fig.update_layout(
        title_text="Combined Line Chart of Adj Close, Close, Open and Volume",
        xaxis_title="Date",
        yaxis_title="Values",
        yaxis2_title="Volume",
        xaxis_range=[start_date, end_date],
        height=600,
        yaxis2=dict(side='right') # Hiển thị trục y phụ ở bên phải
    )

    # Trả về chỉ fig và filtered_df
    return fig, filtered_df


def Analyze_Forecast(symbol, start_date, end_date):
    st.write(f"Analyzing {symbol} from {start_date} to {end_date.strftime('%Y-%m-%d')}")
    df = get_stock_data(symbol, start_date, end_date)
    if df is not None and not df.empty:
        df = df.reset_index()  # Resets index, adding 'Date' column

        st.subheader("Data Source")
        st.dataframe(df)

        # Hiển thị biểu đồ trong st.expander
        with st.expander("Dataset Description"):

            # Tính toán và hiển thị các tham số thống kê
            st.subheader("Statistical Metrics Analysis")
            statistics = calculate_statistics(df)  # Nhận DataFrame từ calculate_statistics
            st.dataframe(statistics)  # Hiển thị DataFrame

            # Tạo bản sao của df và đặt lại index
            df_with_date = df.reset_index()

            # Gọi create_chart và nhận fig
            fig, filtered_df = create_chart(df_with_date, start_date, end_date) # Sử dụng df_with_date
            st.plotly_chart(fig, use_container_width=True)



       # Tạo expander cho bảng tương quan và Pairplot
        with st.expander("Variables correlation"):
            # Nhóm dữ liệu theo ngày và tính toán tổng cho mỗi cột
            daily_data = df.groupby('Date').sum()

            # Loại bỏ cột 'Date' khỏi daily_data vì nó đã trở thành index
            daily_data = daily_data.drop(columns=['Date'], errors='ignore')

            # Tính toán ma trận tương quan
            correlation_matrix = daily_data.corr()  # Sử dụng daily_data

            # Hiển thị bảng tương quan
            st.subheader("Correlation Table")
            st.dataframe(correlation_matrix)
    else:
        st.error(f"No data found for {symbol}")







def create_adj_close_ma_chart_with_prediction(df, ma_window=20, forecast_days=7):
    try:
        # Chuyển đổi Date thành index nếu chưa phải
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # Chuyển đổi 'Adj Close' sang kiểu float
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

        # Loại bỏ các giá trị NaN
        df = df.dropna(subset=['Adj Close'])

        # Tính toán MA
        df['MA'] = df['Adj Close'].rolling(window=ma_window).mean()

        # Tính toán dự đoán
        last_values = df['Adj Close'].tail(ma_window).values
        predictions = []
        for _ in range(forecast_days):
            pred = np.mean(last_values) if len(predictions) == 0 else np.mean(np.append(last_values[1:], predictions[-1]))
            predictions.append(pred)
            last_values = np.append(last_values[1:], pred)

        # Tạo ngày tương lai
        # Đảm bảo last_date là datetime
        last_date = df.index[-1]
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.to_datetime(last_date)

        # Tạo danh sách ngày tương lai
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='B'
        )

        # Tạo biểu đồ
        fig = go.Figure()

        # Đường Adj Close
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Adj Close'],
            mode='lines',
            name='Adj Close',
            line=dict(color='green', width=2),
            opacity=0.8
        ))

        # Đường MA
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA'],
            mode='lines',
            name=f'MA {ma_window}',
            line=dict(color='orange', width=2),
            opacity=0.8
        ))

        # Đường dự đoán
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Prediction',
            line=dict(color='blue', width=2, dash='dash'),
            opacity=0.8
        ))

        # Cập nhật layout
        fig.update_layout(
            title='Adj Close and MA Chart With Prediction',
            xaxis=dict(
                title='Date',
                type='date',
                showgrid=True,
                gridcolor='lightgray',
                tickformat='%Y-%m-%d',
                tickmode='auto',
                nticks=20,
                showline=True,
                linewidth=1,
                linecolor='black',
                range=[df.index[0], future_dates[-1]]
            ),
            yaxis=dict(
                title='Price',
                showgrid=True,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black',
                zeroline=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # DataFrame cho dự đoán
        df_pred = pd.DataFrame({'MA Prediction': predictions}, index=future_dates)

        # Tính metrics
        actual_values = df['Adj Close'].tail(forecast_days).values
        mae = mean_absolute_error(actual_values, predictions[:len(actual_values)])
        rmse = np.sqrt(mean_squared_error(actual_values, predictions[:len(actual_values)]))
        mape = np.mean(np.abs((actual_values - predictions[:len(actual_values)]) / actual_values)) * 100

        return fig, df_pred, mae, rmse, mape

    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
        return None, None, None, None, None


def create_adj_close_holt_chart_with_prediction(df, smoothing_level, beta, forecast_days):
    """
    Creates a line chart for Adj Close and predicts future values using the Holt method.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data with 'Adj Close' column.
        alpha (float, optional): The smoothing parameter for the level (alpha). Defaults to 0.1.
        beta (float, optional): The smoothing parameter for the trend (beta). Defaults to 0.2.
        forecast_days (int, optional): The number of days to forecast. Defaults to 7.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly chart.
        pd.DataFrame: The prediction values in a DataFrame.
    """

    # Chuyển đổi cột 'Date' thành kiểu datetime nếu cần
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Đảm bảo index là DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Tạo và huấn luyện mô hình Holt
    model_holt = Holt(df['Adj Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level, smoothing_trend=beta  # Sử dụng smoothing_level và beta
    )

    # Gán giá trị dự đoán vào cột 'Adj Close Holt'
    df['Adj Close Holt'] = model_holt.fittedvalues

    # Tính toán dự đoán trong mẫu để tính toán lỗi
    du_doan_trong_mau = model_holt.fittedvalues

    # Tính toán các chỉ số lỗi
    mae = mean_absolute_error(df['Adj Close'], du_doan_trong_mau)
    rmse = np.sqrt(mean_squared_error(df['Adj Close'], du_doan_trong_mau))
    mape = np.mean(np.abs((df['Adj Close'] - du_doan_trong_mau) / df['Adj Close'])) * 100

    # Khởi tạo giá trị mức (level) và xu hướng (trend) cuối cùng từ mô hình
    level = model_holt.level[-1]
    trend = model_holt.trend[-1]

    # Dự đoán giá trị tương lai bằng công thức Holt
    predictions = []
    for i in range(forecast_days):
        # Giá trị dự đoán = mức hiện tại + (xu hướng hiện tại * (i + 1))
        prediction = level + (trend * (i + 1))
        predictions.append(prediction)

    # Tạo DataFrame cho dự đoán
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_pred = pd.DataFrame({'Adj Close Holt Prediction': predictions}, index=future_dates)


    # Create line chart
    fig = go.Figure()

    # Adj Close line (Green)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='green')))

    # Holt line for historical data (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Holt'], mode='lines', name='Holt (Historical)', line=dict(color='orange')))

    # Holt Prediction line (Blue)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt Prediction'], mode='lines', name='Holt Prediction', line=dict(color='blue', dash='dash')))

    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[df.index.min(), df.index.max()],
        height=600,
    )


    # Hiển thị tham số và chỉ số lỗi
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {beta:.2f}")
    st.write(f"**Chỉ số lỗi (Holt):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")

    return fig, df_pred


def create_adj_close_holt_winters_chart_with_prediction(df, smoothing_level, smoothing_trend, smoothing_seasonal, seasonality_periods, forecast_days):
    """
    Creates a line chart for Adj Close and predicts future values using the Holt-Winters method.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data with 'Adj Close' column.
        smoothing_level (float): The smoothing parameter for the level (alpha).
        smoothing_trend (float): The smoothing parameter for the trend (beta).
        smoothing_seasonal (float): The smoothing parameter for the seasonality (gamma).
        seasonality_periods (int): The number of periods in a season (e.g., 12 for monthly data with yearly seasonality).
        forecast_days (int): The number of days to forecast.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly chart.
        pd.DataFrame: The prediction values in a DataFrame.
    """
    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Train the Holt-Winters model
    model_hw = ExponentialSmoothing(
        df['Adj Close'],
        trend="add",
        seasonal="add",
        seasonal_periods=seasonality_periods,
        initialization_method="estimated"
    ).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal
    )

    # Add historical predictions to the DataFrame
    df['Adj Close Holt-Winters'] = model_hw.fittedvalues

    # Calculate errors
    mae = mean_absolute_error(df['Adj Close'], df['Adj Close Holt-Winters'])
    rmse = np.sqrt(mean_squared_error(df['Adj Close'], df['Adj Close Holt-Winters']))
    mape = np.mean(np.abs((df['Adj Close'] - df['Adj Close Holt-Winters']) / df['Adj Close'])) * 100

    # Prepare for future predictions
    predictions = []  # Use a list to store predictions

    # Get last values of level and trend
    level = model_hw.level[-1]
    trend = model_hw.trend[-1]

    # Holt-Winters seasonal values are not directly accessible, so we must calculate them
    seasonal_values = model_hw.fittedvalues - (level + trend)

    # Generate predictions for the next forecast_days
    for i in range(forecast_days):
        seasonal_index = (i + len(df)) % seasonality_periods  # Wrap around seasonality
        prediction = level + trend * (i + 1) + seasonal_values[seasonal_index]
        predictions.append(prediction)

    # Create a DataFrame for predictions
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_pred = pd.DataFrame({'Adj Close Holt-Winters Prediction': predictions}, index=future_dates)

    # Create the plot
    fig = go.Figure()

    # Adj Close line (Green)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='green')))

    # Holt-Winters line for historical data (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Holt-Winters'], mode='lines', name='Holt-Winters (Historical)', line=dict(color='orange')))

    # Holt-Winters Prediction line (Blue)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt-Winters Prediction'], mode='lines', name='Holt-Winters Prediction', line=dict(color='blue', dash='dash')))

    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt-Winters Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[df.index.min(), df.index.max()],
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}, Gamma: {smoothing_seasonal:.2f}, Seasonality Periods: {seasonality_periods}")
    st.write(f"**Chỉ số lỗi (Holt Winter):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")

    return fig, df_pred

def apply_holt_monthly(df, smoothing_level, smoothing_trend, forecast_days):
    # Chuyển đổi dữ liệu sang định dạng phù hợp
    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    # Chuyển đổi 'Adj Close' sang kiểu float
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

    # Loại bỏ các giá trị NaN
    df = df.dropna(subset=['Adj Close'])

    # Gom nhóm theo tháng và tính tổng
    monthly_df = df.resample('M')['Adj Close'].mean()
    monthly_df = pd.DataFrame(monthly_df)

    # Train Holt model on monthly data
    model_holt = Holt(monthly_df['Adj Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level, smoothing_trend=smoothing_trend
    )

    # Add historical predictions to DataFrame
    monthly_df['Adj Close Holt'] = model_holt.fittedvalues

    # Get the last level and trend values
    level = model_holt.level[-1]
    trend = model_holt.trend[-1]

    # Generate predictions for the next forecast_days
    predictions = []
    for i in range(forecast_days):
        prediction = level + trend * (i + 1)  # Holt prediction formula
        predictions.append(prediction)

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Adj Close Holt Prediction': predictions}, index=future_dates)

    # Calculate in-sample errors
    mae = mean_absolute_error(monthly_df['Adj Close'], model_holt.fittedvalues)
    rmse = np.sqrt(mean_squared_error(monthly_df['Adj Close'], model_holt.fittedvalues))
    mape = np.mean(np.abs((monthly_df['Adj Close'] - model_holt.fittedvalues) / monthly_df['Adj Close'])) * 100

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close'], mode='lines', name='Adj Close (Historical)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close Holt'], mode='lines', name='Holt (Historical)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt Prediction'], mode='lines', name='Holt Prediction', line=dict(color='blue', dash='dash')))
    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt-Winters Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}%")
    st.write(f"**Chỉ số lỗi (Holt):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")

    return fig, df_pred


def apply_holt_winters_monthly(df, smoothing_level, smoothing_trend, smoothing_seasonal, forecast_days):
    """Applies Holt-Winters method with monthly aggregation and returns predictions."""

    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Aggregate 'Adj Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Adj Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)

    # Train Holt-Winters model on monthly data
    seasonality_periods = 12  # Set to 12 for yearly seasonality with monthly data
    model_hw = ExponentialSmoothing(
        monthly_df['Adj Close'],
        trend="add",
        seasonal="add",
        seasonal_periods=seasonality_periods,
        initialization_method="estimated"
    ).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal
    )

    # Generate future dates for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='MS')  # 'MS' for month start frequency

    # Make predictions
    predictions = model_hw.forecast(forecast_days)

    # Get last values of level and trend
    level = model_hw.level[-1]
    trend = model_hw.trend[-1]

    # Holt-Winters seasonal values are not directly accessible, so we must calculate them
    seasonal_values = model_hw.fittedvalues - (level + trend)

    # Initialize predictions as a list (This is correct)
    predictions = []

    # Add this line to create the 'Adj Close Holt-Winters' column
    monthly_df['Adj Close Holt-Winters'] = model_hw.fittedvalues


    # Generate predictions for the next forecast_days
    for i in range(forecast_days):
        seasonal_index = (i + len(df)) % seasonality_periods  # Wrap around seasonality
        prediction = level + trend * (i + 1) + seasonal_values[seasonal_index]
        predictions.append(prediction)

    # Create DataFrame for predictions
    df_pred = pd.DataFrame({'Adj Close Holt-Winters Prediction': predictions}, index=future_dates)

    # Calculate in-sample errors
    mae = mean_absolute_error(monthly_df['Adj Close'], model_hw.fittedvalues)
    rmse = np.sqrt(mean_squared_error(monthly_df['Adj Close'], model_hw.fittedvalues))
    mape = np.mean(np.abs((monthly_df['Adj Close'] - model_hw.fittedvalues) / monthly_df['Adj Close'])) * 100

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close'], mode='lines', name='Adj Close (Historical)', line=dict(color='green')))

    # Add Holt-Winters historical data line (Orange)
    # Assuming you have the fitted values in a column named 'Adj Close Holt-Winters' in your monthly_df
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close Holt-Winters'], mode='lines', name='Holt-Winters (Historical)', line=dict(color='orange')))

    # Add this trace for future predictions (Blue dashed line)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt-Winters Prediction'], mode='lines', name='Holt-Winters Prediction', line=dict(color='blue', dash='dash')))


    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt-Winters Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}, Gamma: {smoothing_seasonal:.2f}")
    st.write(f"**Chỉ số lỗi (Holt Winter):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")


    return fig, df_pred # R



def main():
    st.set_page_config(
        page_title="Stock Price Prediction Dashboard",
        page_icon="📊",
        layout="wide"
    )

    ma_period = None  # Khởi tạo ma_period bằng None
    # Định nghĩa forecast_days ở đây
    forecast_days = 7  # Hoặc bất kỳ giá trị nào bạn muốn

    # Custom CSS for designing the sticky tab
    st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            position: fixed;
            top: 0;
            right: 0;
            background: #f0f0f0;
            z-index: 100;
            border-bottom: 1px solid #ddd;
        }
        .stTab {
            padding: 1rem;
        }
        .input-section {
            margin-bottom: 2rem;
        }
        .analyze-button {
            margin-top: 1rem;
            padding: 0.5rem 1rem;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .analyze-button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('📊 Stock Price Prediction Dashboard')


    # Create two main tabs in the sidebar
    with st.sidebar:
        st.title("Navigation")  # Sidebar title
        selected_tab = st.radio("Select Tab", ["Statistical Analysis", "Advanced Prediction", "Prediction"])  # Thêm tab "Prediction"


    # Statistical Analysis Tab
    if selected_tab == "Statistical Analysis":
        st.header("Statistical Analysis")
        st.subheader('Input Parameters')

        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input('Stock Symbol', 'AAPL')
        with col2:
            start_date = st.date_input('Start Date', datetime.now() - timedelta(days=1826))
        with col3:
            end_date = st.date_input('End Date', datetime.now())

            # Chuyển đổi start_date và end_date thành chuỗi trước khi lưu trữ
            st.session_state.start_date = start_date.strftime('%Y-%m-%d')
            st.session_state.end_date = end_date.strftime('%Y-%m-%d')

            # Lưu trữ symbol, start_date, end_date vào session_state
            st.session_state.symbol = symbol
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date

            # Khởi tạo start_date và end_date nếu chưa tồn tại
        if 'start_date' not in st.session_state:
            st.session_state.start_date = datetime.now() - timedelta(days=365)
        if 'end_date' not in st.session_state:
            st.session_state.end_date = datetime.now()

           # Thêm xử lý sự kiện cho nút "Analyze"
        if st.button('Analyze'):
            Analyze_Forecast(symbol, start_date, end_date)

    # Advanced Prediction Tab
    elif selected_tab == "Advanced Prediction":
        st.header("✨ Advanced Prediction")

        # Nhập thông tin start_date và end_date
        st.subheader("Enter forecast period:")

        # Danh sách tên các file CSV đã tải sẵn
        csv_files = ["VFC.csv", "TSLA.csv", "NOK.csv", "NKE.csv", "ADDYY.csv"]

        # Lấy danh sách các file CSV trong thư mục dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

        # Tạo lựa chọn file CSV
        selected_file = st.selectbox("Select file CSV", csv_files)

        # Kiểm tra xem người dùng đã chọn file hay chưa
        if selected_file:
            # Đường dẫn tới file CSV đã chọn
            file_path = os.path.join(dataset_dir, selected_file)

            # Đọc dữ liệu từ file CSV đã chọn
            df = pd.read_csv(file_path)

        # Chọn mô hình dự báo
        st.subheader("Select Prediction Model:")
        model_choice = st.selectbox("Model:",
                                    ["Simple Moving Average",
                                    "Holt By Month"
                                    , "Holt Winter By Month",
                                    ])

        # Chọn thời gian dự đoán (chỉ cho Simple Moving Average)
        if model_choice == "Simple Moving Average":
            st.subheader("Select forecast period:")

            # Thêm ô nhập ma_window
            ma_window = st.number_input("Enter MA period:", min_value=1, value=20)

            forecast_period = st.selectbox("Forecast period:",
                                                ["1 day", "1 week (5 days)",
                                                "1 month (22 days)", "Else"])

            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Else":
                custom_days = st.number_input("Enter forecast period:", min_value=1, value=1)
                forecast_days = custom_days  # Gán custom_days cho forecast_days nếu chọn "Khác"
            else:
                # Gán forecast_days dựa trên forecast_period đã chọn
                forecast_days = {
                    "1 day": 1,
                    "1 week (5 days)": 5,
                    "1 month (22 days)": 22,
                }[forecast_period]


        elif model_choice == "Holt By Month":
            st.subheader("Select forecast period:")

            seasonality_periods = st.number_input("Seasonality Periods", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Forecast period:",
                                                ["1 month", "6 month",
                                                "12 month", "Else"])

            # Add sliders for Holt-Winters parameters
            st.subheader("Holt-Winters Parameters")
            alpha_holt = st.slider("Smoothing Level (Alpha)", 0.01, 1.0, 0.2, 0.01)
            beta_holt = st.slider("Smoothing Trend (Beta)", 0.01, 1.0, 0.1, 0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Else":
                custom_days = st.number_input("Enter forecast period:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 month": 1,
                    "6 month": 6,
                    "12 month": 12,
                }[forecast_period]



        elif model_choice == "Holt Winter By Month":
            st.subheader("Forecast period (Month):")

            seasonality_periods = st.number_input("Seasonality Periods", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Forecast period:",
                                                ["1 month", "6 month",
                                                "12 month", "Else"])

            # Add sliders for Holt-Winters parameters
            st.subheader("Holt-Winters Parameters")
            alpha_hwm = st.slider("Smoothing Level (Alpha)", 0.01, 1.0, 0.2, 0.01)
            beta_hwm = st.slider("Smoothing Trend (Beta)", 0.01, 1.0, 0.1, 0.01)
            gamma_hwm = st.slider("Smoothing Seasonal (Gamma)", 0.01, 1.0, 0.1, 0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Else":
                custom_days = st.number_input("Enter forecast period:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 month": 1,
                    "6 month": 6,
                    "12 month": 12,
                }[forecast_period]

        # Nút Dự báo
        if st.button('Predict'):
          if selected_file:
            # Đọc file CSV
            file_path = os.path.join(dataset_dir, selected_file)
            df = pd.read_csv(file_path)

            # Tiền xử lý dữ liệu
            df = preprocess_stock_data(df)


            # Xử lý dự đoán dựa trên model_choice
            if model_choice == "Simple Moving Average":
                # Vẽ biểu đồ SMA (Adj Close và MA) với dự đoán
                # custom_days is defined within your 'Predict' section - ensure it is defined
                      df = pd.read_csv(file_path)

                      # Tạo biểu đồ và dự đoán
                      fig_ma, df_pred, mae, rmse, mape = create_adj_close_ma_chart_with_prediction(
                          df,
                          ma_window=ma_window,
                          forecast_days=forecast_days
                      )

                      if fig_ma is not None:
                          # Hiển thị chỉ số lỗi
                          st.write(f"**Chỉ số lỗi (MA):**")
                          st.write(f"  - MAE: {mae:.2f}")
                          st.write(f"  - RMSE: {rmse:.2f}")
                          st.write(f"  - MAPE: {mape:.2f}%")

                          # Hiển thị biểu đồ
                          st.plotly_chart(fig_ma, use_container_width=True)

                          # Hiển thị bảng dự đoán
                          st.subheader("Prediction Table:")
                          st.dataframe(df_pred)


            elif model_choice == "Holt By Month":
            # Call the Holt-Winters monthly function
                fig_holt_monthly, df_pred_holt_monthly = apply_holt_monthly(
                    df,
                    smoothing_level=alpha_holt,
                    smoothing_trend=beta_holt,
                    forecast_days=ma_period
                )

                # Display the chart and prediction table
                st.plotly_chart(fig_holt_monthly, use_container_width=True)
                st.subheader("Holt (Monthly)Prediction Table:")
                st.dataframe(df_pred_holt_monthly)


            elif model_choice == "Holt Winter By Month":
            # Call the Holt-Winters monthly function
                fig_hwm, df_pred_hwm = apply_holt_winters_monthly(
                    df,
                    smoothing_level=alpha_hwm,
                    smoothing_trend=beta_hwm,
                    smoothing_seasonal=gamma_hwm,
                    forecast_days=ma_period
                )

                # Display the chart and prediction table
                st.plotly_chart(fig_hwm, use_container_width=True)
                st.subheader(" Holt-Winters (Monthly)Prediction Table:")
                st.dataframe(df_pred_hwm)

            else:
                st.warning("Please choose a file CSV.")

    # Price Prediction Tab
    elif selected_tab == "Prediction":
        st.header("📈 Advanced Stock Price Prediction")
        settings = add_settings_sidebar()

        # Input Parameters section at the top
        st.subheader('Input Parameters')

        # Divide the layout into 3 columns for input
        col1, col2, col3 = st.columns(3)

        with col1:
            symbol = st.text_input('Stock Symbol', 'AAPL')

        with col2:
            start_date = st.date_input('Start Date', datetime.now() - timedelta(days=758))

        with col3:
            forecast_days = st.slider('Forecast Days', 1, 30, 7, help="Number of days to forecast")

          # Generate Forecast button

        if st.button('Generate Forecast', use_container_width=True):
            with st.spinner('Loading data...'):
                df = get_stock_data(symbol, start_date)
                if df is not None and not df.empty:
                    df = calculate_technical_indicators(df)
                    predictions = predict_prices(df, forecast_days)

                    if predictions:
                        future_dates = pd.date_range(
                            start=df.index[-1] + pd.Timedelta(days=1),
                            periods=forecast_days
                        )
                        metrics = calculate_metrics(df, predictions, forecast_days)

                        display_enhanced_metrics(metrics)

                        chart_container = st.container()
                        with chart_container:


                            if len(settings['indicators']) > 0:
                                st.subheader('Technical Analysis')
                                for indicator in settings['indicators']:
                                    if indicator == 'MACD':
                                        macd_fig = create_macd_chart(df, symbol)
                                        st.plotly_chart(macd_fig, use_container_width=True)
                                    elif indicator == 'RSI':
                                        rsi_fig = create_rsi_chart(df, symbol)
                                        st.plotly_chart(rsi_fig, use_container_width=True)

                            display_prediction_table(future_dates, predictions, metrics)
                    else:
                        st.error("Failed to generate predictions")
                else:
                    st.error(f"No data found for {symbol}")

if __name__ == "__main__":
    main()

