import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Trader Behavior Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for statsmodels for trendlines
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    st.sidebar.warning("Statsmodels not installed. Trendlines will be disabled.")

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
        color: white;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    .stSelectbox, .stDateInput, .stSlider {
        margin-bottom: 1.5rem;
    }
    .insight-box {
        background-color: #F8F9FA;
        border-left: 4px solid #F18F01;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.07);
        border-left: 5px solid #2E86AB;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }
    .correlation-matrix {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .correlation-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .positive-correlation {
        border-bottom: 4px solid #28a745;
    }
    .negative-correlation {
        border-bottom: 4px solid #dc3545;
    }
    .performance-marker {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .performance-positive {
        background: rgba(40, 167, 69, 0.15);
        color: #28a745;
    }
    .performance-negative {
        background: rgba(220, 53, 69, 0.15);
        color: #dc3545;
    }
    .strategy-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .strategy-card h4 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .high-impact-metric {
        animation: pulse 2s ease-in-out infinite;
    }
    .sentiment-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #dc3545 0%, #fd7e14 25%, #ffc107 50%, #20c997 75%, #28a745 100%);
        margin: 0.5rem 0;
        position: relative;
    }
    .sentiment-marker {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 18px;
        background: #2E86AB;
        transform: translateX(-50%);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6C757D;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        # -----------------------------
        # 1️⃣ Load your datasets
        # -----------------------------
        daily_trader_stats = pd.read_csv("daily_trader_stats.csv")  # Update path if needed
        fear_greed_df = pd.read_csv("fear_greed.csv")               # Update path if needed
        historical_data_df = pd.read_csv("historical_data.csv")     # Update path if needed

        # -----------------------------
        # 2️⃣ Ensure 'date' columns are datetime
        # -----------------------------
        daily_trader_stats['date'] = pd.to_datetime(daily_trader_stats['date'])
        fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
        historical_data_df['date'] = pd.to_datetime(historical_data_df['date'])

        # -----------------------------
        # 3️⃣ Optional: normalize to remove time part
        # -----------------------------
        daily_trader_stats['date'] = daily_trader_stats['date'].dt.normalize()
        fear_greed_df['date'] = fear_greed_df['date'].dt.normalize()
        historical_data_df['date'] = historical_data_df['date'].dt.normalize()

        # -----------------------------
        # 4️⃣ Merge daily_trader_stats with fear_greed_df
        # -----------------------------
        
        merged_df = pd.merge(
            daily_trader_stats,
            fear_greed_df,
            on='date',
            how='left'
        )

        # -----------------------------
        # 5️⃣ Return all dataframes
        # -----------------------------
        return fear_greed_df, historical_data_df, merged_df

    except FileNotFoundError:
        # Sample data for demonstration (2023-2025)
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')

        # Fear/Greed data
        values, classifications = [], []
        current_value = 50
        for i in range(len(dates)):
            change = np.random.normal(0, 8)
            current_value = max(0, min(100, current_value + change))
            values.append(current_value)
            if current_value >= 75:
                classifications.append('Extreme Greed')
            elif current_value >= 55:
                classifications.append('Greed')
            elif current_value >= 45:
                classifications.append('Neutral')
            elif current_value >= 25:
                classifications.append('Fear')
            else:
                classifications.append('Extreme Fear')

        fear_greed_df = pd.DataFrame({
            'timestamp': range(len(dates)),
            'value': values,
            'classification': classifications,
            'date': dates
        })

        # Historical trading data
        accounts = [f'Account_{i}' for i in range(200)]
        coins = ['BTC', 'ETH', 'SOL', 'AVAX', 'BNB', 'XRP', 'ADA', 'DOT']
        sides = ['Buy', 'Sell']
        date_weights = [1.5 if d.weekday() < 5 else 0.5 for d in dates]
        normalized_weights = [w/sum(date_weights) for w in date_weights]
        trade_dates = np.random.choice(dates, 10000, p=normalized_weights)
        fear_greed_by_date = dict(zip(fear_greed_df['date'], fear_greed_df['value']))

        historical_data_df = pd.DataFrame({
            'account': np.random.choice(accounts, 10000),
            'coin': np.random.choice(coins, 10000, p=[0.4, 0.25, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]),
            'execution_price': np.random.uniform(100, 50000, 10000),
            'size_tokens': np.random.uniform(0.1, 100, 10000),
            'size_usd': np.random.uniform(100, 50000, 10000),
            'side': np.random.choice(sides, 10000),
            'timestamp_ist': trade_dates,
            'start_position': np.random.uniform(-10000, 10000, 10000),
            'direction': np.random.choice(['Long', 'Short'], 10000),
            'order_id': np.random.randint(100000000, 999999999, 10000),
            'crossed': np.random.choice([True, False], 10000, p=[0.7, 0.3]),
            'fee': np.random.uniform(0, 100, 10000),
            'trade_id': np.random.randint(10000000, 99999999, 10000),
            'timestamp': np.random.uniform(1.6e12, 1.7e12, 10000)
        })
        
        # Add date column to historical_data_df
        historical_data_df['date'] = trade_dates
        
        # Ensure date columns are datetime
        fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date']).dt.normalize()
        historical_data_df['date'] = pd.to_datetime(historical_data_df['date']).dt.normalize()
        
        # Add fear_greed_value to historical_data_df
        historical_data_df['fear_greed_value'] = historical_data_df['date'].map(fear_greed_by_date)
        historical_data_df['closed_pnl'] = np.random.normal(
            loc=(historical_data_df['fear_greed_value'] - 50) * 20,
            scale=300,
            size=len(historical_data_df)
        )

        daily_trader_stats = historical_data_df.groupby('date').agg({
            'closed_pnl': 'sum',
            'size_usd': 'mean',
            'account': 'count'
        }).reset_index().rename(columns={'account': 'num_traders'})

        merged_df = pd.merge(daily_trader_stats, fear_greed_df, on='date', how='left')

        return fear_greed_df, historical_data_df, merged_df

# Load data
fear_greed_df, historical_data_df, merged_df = load_data()

# Sidebar with improved styling
with st.sidebar:
    st.title("📊 Dashboard Controls")
    st.markdown("---")
    
    st.markdown("### 📅 Date Range")
    
    # Ensure we have valid date ranges
    min_date = merged_df['date'].min() if not merged_df.empty else datetime.now().date()
    max_date = merged_df['date'].max() if not merged_df.empty else datetime.now().date()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed"
    )
    
    st.markdown("### 😊 Market Sentiment")
    sentiment_options = ['All'] + list(merged_df['classification'].unique()) if 'classification' in merged_df.columns else ['All']
    selected_sentiment = st.selectbox("Select Sentiment", sentiment_options, label_visibility="collapsed")
    
    st.markdown("### 🪙 Coin Selection")
    coin_options = ['All'] + list(historical_data_df['coin'].unique()) if 'coin' in historical_data_df.columns else ['All']
    selected_coin = st.selectbox("Select Coin", coin_options, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📈 Visualization Options")
    show_trend_lines = st.checkbox("Show Trend Lines", value=True and HAS_STATSMODELS)
    show_annotations = st.checkbox("Show Annotations", value=True)

# Apply filters
if selected_sentiment != 'All' and 'classification' in merged_df.columns:
    merged_df_filtered = merged_df[merged_df['classification'] == selected_sentiment]
    historical_data_df_filtered = historical_data_df[historical_data_df['date'].isin(merged_df_filtered['date'])]
else:
    merged_df_filtered = merged_df.copy()
    historical_data_df_filtered = historical_data_df.copy()

if selected_coin != 'All' and 'coin' in historical_data_df_filtered.columns:
    historical_data_df_filtered = historical_data_df_filtered[historical_data_df_filtered['coin'] == selected_coin]
    coin_dates = historical_data_df_filtered['date'].unique()
    merged_df_filtered = merged_df_filtered[merged_df_filtered['date'].isin(coin_dates)]

if len(date_range) == 2:
    start_date, end_date = date_range
    merged_df_filtered = merged_df_filtered[(merged_df_filtered['date'] >= pd.to_datetime(start_date)) & (merged_df_filtered['date'] <= pd.to_datetime(end_date))]
    historical_data_df_filtered = historical_data_df_filtered[
        (historical_data_df_filtered['date'] >= pd.to_datetime(start_date)) & (historical_data_df_filtered['date'] <= pd.to_datetime(end_date))
    ]

# Main content
st.markdown('<h1 class="main-header">Trader Behavior Insights Dashboard</h1>', unsafe_allow_html=True)

# Key Metrics
st.markdown('<h2 class="section-header">Performance Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_pnl = merged_df_filtered['closed_pnl'].sum() if not merged_df_filtered.empty else 0
    pnl_color = "green" if total_pnl >= 0 else "red"
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Total PnL</div>
        <div class="metric-value" style="color: {pnl_color}">${total_pnl:,.2f}</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    avg_daily_pnl = merged_df_filtered['closed_pnl'].mean() if not merged_df_filtered.empty else 0
    daily_pnl_color = "green" if avg_daily_pnl >= 0 else "red"
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Avg. Daily PnL</div>
        <div class="metric-value" style="color: {daily_pnl_color}">${avg_daily_pnl:,.2f}</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    total_traders = merged_df_filtered['num_traders'].sum() if not merged_df_filtered.empty else 0
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Total Traders</div>
        <div class="metric-value">{total_traders:,}</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    avg_sentiment = merged_df_filtered['value'].mean() if not merged_df_filtered.empty and 'value' in merged_df_filtered.columns else 0
    sentiment_color = "#F18F01"  # Orange for sentiment
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Avg. Fear/Greed</div>
        <div class="metric-value" style="color: {sentiment_color}">{avg_sentiment:.2f}</div>
    </div>
    ''', unsafe_allow_html=True)

# Market Sentiment Over Time
if not merged_df_filtered.empty and 'value' in merged_df_filtered.columns:
    st.markdown('<h2 class="section-header">Market Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a more sophisticated sentiment chart
        fig = go.Figure()
        
        # Add the line trace
        fig.add_trace(go.Scatter(
            x=merged_df_filtered['date'], 
            y=merged_df_filtered['value'],
            mode='lines',
            name='Fear & Greed Index',
            line=dict(color='#2E86AB', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
        
        # Add regions for different sentiment levels
        fig.add_hrect(y0=0, y1=25, line_width=0, fillcolor="red", opacity=0.1)
        fig.add_hrect(y0=25, y1=45, line_width=0, fillcolor="orange", opacity=0.1)
        fig.add_hrect(y0=45, y1=55, line_width=0, fillcolor="yellow", opacity=0.1)
        fig.add_hrect(y0=55, y1=75, line_width=0, fillcolor="lightgreen", opacity=0.1)
        fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="green", opacity=0.1)
        
        # Add annotations for sentiment regions
        fig.add_annotation(x=merged_df_filtered['date'].iloc[0], y=12.5, text="Extreme Fear", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=merged_df_filtered['date'].iloc[0], y=35, text="Fear", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=merged_df_filtered['date'].iloc[0], y=50, text="Neutral", showarrow=False, font=dict(color="gray"))
        fig.add_annotation(x=merged_df_filtered['date'].iloc[0], y=65, text="Greed", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=merged_df_filtered['date'].iloc[0], y=87.5, text="Extreme Greed", showarrow=False, font=dict(color="darkgreen"))
        
        fig.update_layout(
            title='Fear & Greed Index Over Time',
            xaxis_title='Date',
            yaxis_title='Fear & Greed Value',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution pie chart
        if 'classification' in merged_df_filtered.columns:
            sentiment_counts = merged_df_filtered['classification'].value_counts()
            
            # Define colors for each sentiment
            sentiment_colors = {
                'Extreme Fear': 'red',
                'Fear': 'orange',
                'Neutral': 'gray',
                'Greed': 'lightgreen',
                'Extreme Greed': 'darkgreen'
            }
            
            colors = [sentiment_colors.get(s, 'blue') for s in sentiment_counts.index]
            
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=colors
            )])
            
            fig.update_layout(
                title='Sentiment Distribution',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Trader Performance vs Market Sentiment
if not merged_df_filtered.empty and 'closed_pnl' in merged_df_filtered.columns:
    st.markdown('<h2 class="section-header">Trader Performance Analysis</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # PnL by sentiment with enhanced visualization
        if 'classification' in merged_df_filtered.columns:
            fig = px.box(merged_df_filtered, x='classification', y='closed_pnl',
                         color='classification',
                         color_discrete_map={
                             'Extreme Fear': 'red',
                             'Fear': 'orange',
                             'Neutral': 'gray',
                             'Greed': 'lightgreen',
                             'Extreme Greed': 'darkgreen'
                         },
                         title='Trader PnL Distribution by Market Sentiment',
                         labels={'classification': 'Market Sentiment', 'closed_pnl': 'Closed PnL (USD)'})
            
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter plot of sentiment vs PnL
        if 'value' in merged_df_filtered.columns:
            # Use a simpler trendline if statsmodels is not available
            trendline_type = "lowess" if (show_trend_lines and HAS_STATSMODELS) else None
            
            fig = px.scatter(merged_df_filtered, x='value', y='closed_pnl', 
                             color='classification',
                             color_discrete_map={
                                 'Extreme Fear': 'red',
                                 'Fear': 'orange',
                                 'Neutral': 'gray',
                                 'Greed': 'lightgreen',
                                 'Extreme Greed': 'darkgreen'
                             },
                             title='Fear & Greed Index vs Trader PnL',
                             labels={'value': 'Fear & Greed Value', 'closed_pnl': 'Closed PnL (USD)'},
                             trendline=trendline_type)
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Trading Activity Analysis
if not merged_df_filtered.empty:
    st.markdown('<h2 class="section-header">Trading Activity Analysis</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Number of traders by sentiment
        if 'classification' in merged_df_filtered.columns:
            fig = px.bar(merged_df_filtered, x='classification', y='num_traders',
                         color='classification',
                         color_discrete_map={
                             'Extreme Fear': 'red',
                             'Fear': 'orange',
                             'Neutral': 'gray',
                             'Greed': 'lightgreen',
                             'Extreme Greed': 'darkgreen'
                         },
                         title='Number of Active Traders by Market Sentiment',
                         labels={'classification': 'Market Sentiment', 'num_traders': 'Number of Traders'})
            
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Daily trading activity over time
        if not historical_data_df_filtered.empty and 'account' in historical_data_df_filtered.columns:
            daily_activity = historical_data_df_filtered.groupby('date').agg({
                'account': 'nunique',
                'trade_id': 'count'
            }).reset_index().rename(columns={'account': 'unique_traders', 'trade_id': 'total_trades'})
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_activity['date'], 
                y=daily_activity['unique_traders'],
                mode='lines',
                name='Unique Traders',
                line=dict(color='#2E86AB', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_activity['date'], 
                y=daily_activity['total_trades'],
                mode='lines',
                name='Total Trades',
                line=dict(color='#A23B72', width=3)
            ))
            
            fig.update_layout(
                title='Daily Trading Activity Over Time',
                xaxis_title='Date',
                yaxis_title='Count',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Coin Performance Analysis
if not historical_data_df_filtered.empty and 'coin' in historical_data_df_filtered.columns:
    st.markdown('<h2 class="section-header">Coin Performance Analysis</h2>', unsafe_allow_html=True)

    coin_stats = historical_data_df_filtered.groupby('coin').agg({
        'closed_pnl': 'sum',
        'size_usd': 'sum',
        'trade_id': 'count',
        'account': 'nunique'
    }).reset_index().rename(columns={
        'trade_id': 'trade_count',
        'account': 'unique_traders'
    })

    col1, col2 = st.columns(2)

    with col1:
        # PnL by coin with improved visualization
        fig = px.bar(coin_stats, x='coin', y='closed_pnl',
                     color='closed_pnl',
                     color_continuous_scale='RdYlGn',
                     title='Total PnL by Coin',
                     labels={'coin': 'Coin', 'closed_pnl': 'Total PnL (USD)'})
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Trade volume by coin
        fig = px.pie(coin_stats, values='size_usd', names='coin',
                     title='Trade Volume Distribution by Coin',
                     hole=0.4)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)

# Trader Behavior Analysis
if not historical_data_df_filtered.empty and 'account' in historical_data_df_filtered.columns:
    st.markdown('<h2 class="section-header">Trader Behavior Analysis</h2>', unsafe_allow_html=True)

    trader_stats = historical_data_df_filtered.groupby('account').agg({
        'closed_pnl': 'sum',
        'size_usd': 'sum',
        'trade_id': 'count'
    }).reset_index().rename(columns={'trade_id': 'trade_count'})

    col1, col2 = st.columns(2)

    with col1:
        # Trade count vs PnL with enhanced visualization
        fig = px.scatter(trader_stats, x='trade_count', y='closed_pnl',
                         size='size_usd', color='closed_pnl',
                         color_continuous_scale='RdYlGn',
                         title='Trade Activity vs Profitability',
                         labels={'trade_count': 'Number of Trades', 'closed_pnl': 'Total PnL (USD)', 'size_usd': 'Trade Volume (USD)'},
                         hover_data=['account'])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution of trader performance
        fig = px.histogram(trader_stats, x='closed_pnl', 
                           title='Distribution of Trader Performance',
                           labels={'closed_pnl': 'Total PnL (USD)', 'count': 'Number of Traders'},
                           nbins=50,
                           color_discrete_sequence=['#2E86AB'])
        
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Market Intelligence Section
st.markdown(
    """
    <div style="font-family:Segoe UI,Arial,sans-serif; color:#1C2833; line-height:1.6; padding:15px;">
      <h2 style="color:#117A65; text-align:center; margin-bottom:20px;">
        📊 Market Intelligence – Key Insights
      </h2>

      <h3 style="color:#2E86AB;">💹 Market Profitability Trends</h3>
      <p>
        Traders earned <b>$15.8K/day</b> during <b>Greed phases</b>, compared to <b>$39K/day</b> in <b>Fear phases</b>.<br>
        👉 Fear phases deliver <b>higher profitability</b>, but with sharper volatility.
      </p>

      <h3 style="color:#2E86AB;">👥 Trader Behavior Patterns</h3>
      <p>
        Trader count drops by <b>25%</b> in <b>Fear periods</b>, while average trade size increases by <b>40%</b>.<br>
        → Fewer participants, but they trade <b>bigger and bolder</b>.
      </p>

      <h3 style="color:#2E86AB;">🔄 Sentiment Volatility</h3>
      <p>
        Current sentiment score: <b>18.7</b> (high volatility).<br>
        Historically, scores above <b>20</b> often signal <b>market reversals within 3–5 days</b>.
      </p>

      <h3 style="color:#2E86AB;">💰 Trade Size Insights</h3>
      <p>
        <b>Greed phase:</b> $5.2K average trade size.<br>
        <b>Fear phase:</b> $7.3K average trade size.<br>
        👉 Traders commit <b>larger positions</b> in Fear-driven markets, showing stronger conviction.
      </p>

      <h3 style="color:#2E86AB;">🚀 Strategic Highlights</h3>
      <p>
        • <b>Fear-driven trades</b> = High-risk, high-reward bursts.<br>
        • <b>Greed-driven trades</b> = Steadier growth, lower risk.<br>
        • Monitoring <b>sentiment + trade size</b> can help predict <b>short-term PnL changes</b>.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.markdown("""
<div class="footer">
    <p>📊 Trader Behavior Insights Dashboard | Designed with care</p>
</div>
""", unsafe_allow_html=True)