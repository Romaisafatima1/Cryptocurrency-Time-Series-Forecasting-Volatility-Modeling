#!/usr/bin/env python3
"""
Interactive Cryptocurrency Forecasting Dashboard
Week 9: Dashboard Development - Complete Implementation
Uses Dash/Plotly for real-time interactive visualization
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import json

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Crypto Forecasting Dashboard",
    update_title="Loading...",
    suppress_callback_exceptions=True
)

# Color scheme
COLORS = {
    'background': '#F8F9FA',
    'card': '#FFFFFF',
    'primary': '#2C3E50',
    'btc': '#F7931A',
    'eth': '#627EEA',
    'success': '#27AE60',
    'warning': '#E67E22',
    'danger': '#E74C3C',
    'text': '#2C3E50',
    'text_light': '#7F8C8D',
    'border': '#BDC3C7'
}

# API configuration
API_BASE_URL = "http://localhost:8000"

# Dashboard layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🔮 Cryptocurrency Forecasting Dashboard", 
                   style={'color': COLORS['primary'], 'marginBottom': '10px'}),
            html.P("Advanced Time Series Analysis & Volatility Modeling",
                  style={'color': COLORS['text_light'], 'fontSize': '16px'})
        ], style={'flex': '1'}),
        
        html.Div([
            html.Div([
                html.Span("🟢 ", style={'fontSize': '20px'}),
                html.Span("API Connected", id='api-status', 
                         style={'color': COLORS['success'], 'fontWeight': 'bold'})
            ]),
            html.Div(id='last-update', 
                    style={'color': COLORS['text_light'], 'fontSize': '12px'})
        ], style={'textAlign': 'right'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '20px 30px',
        'backgroundColor': COLORS['card'],
        'borderBottom': f'3px solid {COLORS["btc"]}',
        'marginBottom': '20px'
    }),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Cryptocurrency", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='crypto-dropdown',
                options=[
                    {'label': '₿ Bitcoin (BTC)', 'value': 'BTC'},
                    {'label': 'Ξ Ethereum (ETH)', 'value': 'ETH'}
                ],
                value='BTC',
                clearable=False,
                style={'width': '200px'}
            )
        ], style={'marginRight': '20px'}),
        
        html.Div([
            html.Label("Model", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'ARIMA (Price)', 'value': 'ARIMA'},
                    {'label': 'LSTM (Price)', 'value': 'LSTM'},
                    {'label': 'GARCH (Volatility)', 'value': 'GARCH'},
                    {'label': 'EGARCH (Volatility)', 'value': 'EGARCH'}
                ],
                value='ARIMA',
                clearable=False,
                style={'width': '200px'}
            )
        ], style={'marginRight': '20px'}),
        
        html.Div([
            html.Label("Forecast Days", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Slider(
                id='forecast-slider',
                min=3,
                max=30,
                step=1,
                value=7,
                marks={3: '3d', 7: '7d', 14: '14d', 21: '21d', 30: '30d'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'flex': '1', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Confidence Level", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Slider(
                id='confidence-slider',
                min=80,
                max=99,
                step=1,
                value=95,
                marks={80: '80%', 90: '90%', 95: '95%', 99: '99%'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '250px', 'marginRight': '20px'}),
        
        html.Button(
            '🔄 Generate Forecast',
            id='generate-button',
            n_clicks=0,
            style={
                'padding': '10px 30px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'backgroundColor': COLORS['btc'],
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'alignSelf': 'flex-end'
            }
        )
    ], style={
        'display': 'flex',
        'alignItems': 'flex-end',
        'padding': '20px',
        'backgroundColor': COLORS['card'],
        'borderRadius': '10px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Main Content Area
    html.Div([
        # Left Column - Main Chart
        html.Div([
            dcc.Loading(
                id="loading-main-chart",
                type="circle",
                children=[
                    dcc.Graph(
                        id='main-forecast-chart',
                        style={'height': '500px'},
                        config={'displayModeBar': True, 'displaylogo': False}
                    )
                ]
            )
        ], style={
            'flex': '2',
            'backgroundColor': COLORS['card'],
            'borderRadius': '10px',
            'padding': '20px',
            'marginRight': '20px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # Right Column - Metrics
        html.Div([
            html.H3("📊 Model Performance", style={'color': COLORS['primary'], 'marginBottom': '20px'}),
            html.Div(id='metrics-panel'),
            
            html.Hr(style={'margin': '20px 0', 'border': f'1px solid {COLORS["border"]}'}),
            
            html.H3("🏆 Best Models", style={'color': COLORS['primary'], 'marginBottom': '20px'}),
            html.Div(id='best-models-panel')
        ], style={
            'flex': '1',
            'backgroundColor': COLORS['card'],
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'maxHeight': '540px',
            'overflowY': 'auto'
        })
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Tabs Section
    html.Div([
        dcc.Tabs(id='analysis-tabs', value='performance', children=[
            dcc.Tab(label='📈 Performance Metrics', value='performance', 
                   style={'padding': '10px', 'fontWeight': 'bold'}),
            dcc.Tab(label='🔬 Residual Diagnostics', value='residuals',
                   style={'padding': '10px', 'fontWeight': 'bold'}),
            dcc.Tab(label='📊 Volatility Analysis', value='volatility',
                   style={'padding': '10px', 'fontWeight': 'bold'}),
            dcc.Tab(label='🔄 Model Comparison', value='comparison',
                   style={'padding': '10px', 'fontWeight': 'bold'})
        ]),
        html.Div(id='tab-content', style={'padding': '20px'})
    ], style={
        'backgroundColor': COLORS['card'],
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    }),
    
    # Footer
    html.Div([
        html.P("🔮 Cryptocurrency Forecasting System | Models: ARIMA, LSTM, GARCH, EGARCH", 
              style={'color': COLORS['text_light'], 'textAlign': 'center', 'margin': '0'})
    ], style={
        'padding': '20px',
        'backgroundColor': COLORS['card'],
        'borderTop': f'1px solid {COLORS["border"]}'
    }),
    
    # Hidden div for storing data
    html.Div(id='forecast-data-store', style={'display': 'none'})
    
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '0'})

# Callback for generating forecast
@app.callback(
    [Output('main-forecast-chart', 'figure'),
     Output('metrics-panel', 'children'),
     Output('best-models-panel', 'children'),
     Output('forecast-data-store', 'children'),
     Output('last-update', 'children'),
     Output('api-status', 'children'),
     Output('api-status', 'style')],
    [Input('generate-button', 'n_clicks')],
    [State('crypto-dropdown', 'value'),
     State('model-dropdown', 'value'),
     State('forecast-slider', 'value'),
     State('confidence-slider', 'value')]
)
def update_forecast(n_clicks, crypto, model, forecast_days, confidence):
    """Generate and display forecast"""
    if n_clicks == 0:
        # Initial load - create empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Select parameters and click 'Generate Forecast' to begin",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=500
        )
        return (fig, 
                html.P("No data yet", style={'color': COLORS['text_light']}),
                html.P("No data yet", style={'color': COLORS['text_light']}),
                json.dumps({}),
                "Never updated",
                "API Connected",
                {'color': COLORS['success'], 'fontWeight': 'bold'})
    
    try:
        # Call API
        response = requests.post(
            f"{API_BASE_URL}/forecast",
            json={
                "cryptocurrency": crypto,
                "forecast_days": forecast_days,
                "confidence_level": confidence / 100,
                "models": [model]
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API returned status {response.status_code}")
        
        data = response.json()
        
        # Create main chart
        fig = create_forecast_chart(data, model, crypto, confidence)
        
        # Create metrics panel
        metrics = create_metrics_panel(data, model)
        
        # Create best models panel
        best_models = create_best_models_panel(data)
        
        # Update timestamp
        timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return (fig, metrics, best_models, json.dumps(data), timestamp,
                "API Connected", {'color': COLORS['success'], 'fontWeight': 'bold'})
    
    except Exception as e:
        # Error handling
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: {str(e)}",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=500
        )
        
        error_msg = html.Div([
            html.P(f"❌ Error: {str(e)}", style={'color': COLORS['danger']})
        ])
        
        return (fig, error_msg, error_msg, json.dumps({}),
                f"Error at {datetime.now().strftime('%H:%M:%S')}",
                "API Error", {'color': COLORS['danger'], 'fontWeight': 'bold'})

def create_forecast_chart(data, model, crypto, confidence):
    """Create the main forecast chart"""
    fig = go.Figure()
    
    # Get forecast data
    if model in ['ARIMA', 'LSTM']:
        forecast = data['price_forecast'].get(model, [])
        confidence_intervals = data['price_confidence_intervals'].get(model, [])
        y_title = "Price (USD)"
        chart_title = f"{crypto} Price Forecast - {model} Model"
    else:
        forecast = data['volatility_forecast'].get(model, [])
        confidence_intervals = []
        y_title = "Volatility (%)"
        chart_title = f"{crypto} Volatility Forecast - {model} Model"
    
    x_values = list(range(1, len(forecast) + 1))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color=COLORS['btc'] if crypto == 'BTC' else COLORS['eth'], width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence intervals if available
    if confidence_intervals:
        upper = [ci['upper'] for ci in confidence_intervals]
        lower = [ci['lower'] for ci in confidence_intervals]
        
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence}% Confidence Interval',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': chart_title,
            'font': {'size': 20, 'color': COLORS['primary'], 'family': 'Arial Bold'}
        },
        xaxis_title="Forecast Day",
        yaxis_title=y_title,
        template="plotly_white",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_metrics_panel(data, model):
    """Create metrics display panel"""
    metrics = data['model_metrics'].get(model, {})
    
    if not metrics:
        return html.P("No metrics available", style={'color': COLORS['text_light']})
    
    metric_cards = []
    
    # Define metrics to display based on model type
    if model in ['ARIMA', 'LSTM']:
        metric_defs = [
            ('RMSE', 'rmse', '📏', COLORS['warning']),
            ('MAE', 'mae', '📊', COLORS['success']),
            ('MAPE', 'mape', '🎯', COLORS['danger'])
        ]
    else:
        metric_defs = [
            ('AIC', 'aic', '📈', COLORS['warning']),
            ('BIC', 'bic', '📉', COLORS['success']),
            ('RMSE', 'rmse', '📏', COLORS['danger'])
        ]
    
    for label, key, icon, color in metric_defs:
        if key in metrics:
            value = metrics[key]
            if key == 'mape':
                display_value = f"{value:.2f}%"
            elif key in ['aic', 'bic']:
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:.2f}"
            
            metric_cards.append(
                html.Div([
                    html.Div([
                        html.Span(icon, style={'fontSize': '24px', 'marginRight': '10px'}),
                        html.Span(label, style={'fontSize': '14px', 'color': COLORS['text_light']})
                    ], style={'marginBottom': '5px'}),
                    html.Div(display_value, style={
                        'fontSize': '28px',
                        'fontWeight': 'bold',
                        'color': color
                    })
                ], style={
                    'padding': '15px',
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '8px',
                    'marginBottom': '10px',
                    'border': f'2px solid {color}'
                })
            )
    
    return html.Div(metric_cards)

def create_best_models_panel(data):
    """Create best models display panel"""
    best_models = data.get('best_models', {})
    
    if not best_models:
        return html.P("No data available", style={'color': COLORS['text_light']})
    
    panels = []
    
    if 'price' in best_models:
        panels.append(
            html.Div([
                html.Div([
                    html.Span("💰 ", style={'fontSize': '20px'}),
                    html.Span("Best Price Model", style={'fontWeight': 'bold', 'color': COLORS['text']})
                ], style={'marginBottom': '10px'}),
                html.Div(best_models['price'], style={
                    'fontSize': '24px',
                    'fontWeight': 'bold',
                    'color': COLORS['success'],
                    'padding': '10px',
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '5px',
                    'textAlign': 'center'
                })
            ], style={'marginBottom': '20px'})
        )
    
    if 'volatility' in best_models:
        panels.append(
            html.Div([
                html.Div([
                    html.Span("📊 ", style={'fontSize': '20px'}),
                    html.Span("Best Volatility Model", style={'fontWeight': 'bold', 'color': COLORS['text']})
                ], style={'marginBottom': '10px'}),
                html.Div(best_models['volatility'], style={
                    'fontSize': '24px',
                    'fontWeight': 'bold',
                    'color': COLORS['warning'],
                    'padding': '10px',
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '5px',
                    'textAlign': 'center'
                })
            ])
        )
    
    return html.Div(panels)

# Callback for tab content
@app.callback(
    Output('tab-content', 'children'),
    [Input('analysis-tabs', 'value')],
    [State('forecast-data-store', 'children'),
     State('crypto-dropdown', 'value')]
)
def update_tab_content(tab, forecast_data_json, crypto):
    """Update tab content based on selection"""
    if not forecast_data_json or forecast_data_json == '{}':
        return html.Div([
            html.P("Generate a forecast to view analysis", 
                  style={'textAlign': 'center', 'color': COLORS['text_light'], 'padding': '40px'})
        ])
    
    try:
        data = json.loads(forecast_data_json)
    except:
        return html.Div([html.P("Error loading data")])
    
    if tab == 'performance':
        return create_performance_tab(data, crypto)
    elif tab == 'residuals':
        return create_residuals_tab(data, crypto)
    elif tab == 'volatility':
        return create_volatility_tab(data, crypto)
    elif tab == 'comparison':
        return create_comparison_tab(data)
    
    return html.Div()

def create_performance_tab(data, crypto):
    """Create performance metrics comparison tab"""
    metrics = data.get('model_metrics', {})
    
    if not metrics:
        return html.P("No metrics available")
    
    # Create bar charts for each metric
    models = list(metrics.keys())
    
    # RMSE comparison
    rmse_values = [metrics[m].get('rmse', 0) for m in models]
    mae_values = [metrics[m].get('mae', 0) for m in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE Comparison', 'MAE Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name='RMSE', 
              marker_color=COLORS['btc'] if crypto == 'BTC' else COLORS['eth']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name='MAE',
              marker_color=COLORS['success']),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white",
        title_text=f"{crypto} Model Performance Comparison"
    )
    
    # Create summary table
    table_data = []
    for model, model_metrics in metrics.items():
        row = {'Model': model}
        for key, value in model_metrics.items():
            if isinstance(value, (int, float)):
                row[key.upper()] = f"{value:.2f}"
            else:
                row[key.upper()] = str(value)
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Hr(style={'margin': '20px 0'}),
        html.H4("Detailed Metrics Table", style={'color': COLORS['primary']}),
        html.Div([
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col, style={'padding': '10px', 'textAlign': 'left', 
                                                  'backgroundColor': COLORS['background']}) 
                            for col in df_table.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(df_table.iloc[i][col], 
                               style={'padding': '10px', 'borderBottom': f'1px solid {COLORS["border"]}'}) 
                        for col in df_table.columns
                    ]) for i in range(len(df_table))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ])
    ])

def create_residuals_tab(data, crypto):
    """Create residuals analysis tab"""
    residual_stats = data.get('residual_statistics', {})
    
    if not residual_stats:
        return html.P("Residual analysis only available for ARIMA and LSTM models")
    
    # Create residual statistics display
    stat_cards = []
    
    for model, stats in residual_stats.items():
        card = html.Div([
            html.H4(f"{model} Residuals", style={'color': COLORS['primary'], 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Div("Mean", style={'fontWeight': 'bold', 'color': COLORS['text_light']}),
                    html.Div(f"{stats['mean']:.4f}", style={'fontSize': '20px', 'color': COLORS['text']})
                ], style={'flex': '1', 'padding': '10px'}),
                html.Div([
                    html.Div("Std Dev", style={'fontWeight': 'bold', 'color': COLORS['text_light']}),
                    html.Div(f"{stats['std']:.4f}", style={'fontSize': '20px', 'color': COLORS['text']})
                ], style={'flex': '1', 'padding': '10px'}),
                html.Div([
                    html.Div("Skewness", style={'fontWeight': 'bold', 'color': COLORS['text_light']}),
                    html.Div(f"{stats['skewness']:.4f}", style={'fontSize': '20px', 'color': COLORS['text']})
                ], style={'flex': '1', 'padding': '10px'}),
                html.Div([
                    html.Div("Kurtosis", style={'fontWeight': 'bold', 'color': COLORS['text_light']}),
                    html.Div(f"{stats['kurtosis']:.4f}", style={'fontSize': '20px', 'color': COLORS['text']})
                ], style={'flex': '1', 'padding': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
            
            html.Div([
                html.Div("✅ Residuals should have:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                html.Ul([
                    html.Li(f"Mean ≈ 0 (Current: {stats['mean']:.4f})"),
                    html.Li(f"Skewness ≈ 0 (Current: {stats['skewness']:.4f})"),
                    html.Li(f"Kurtosis ≈ 0 (Current: {stats['kurtosis']:.4f})")
                ])
            ], style={'marginTop': '15px', 'padding': '15px', 'backgroundColor': COLORS['background'], 
                     'borderRadius': '5px'})
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['card'],
            'borderRadius': '10px',
            'marginBottom': '20px',
            'border': f'2px solid {COLORS["border"]}'
        })
        stat_cards.append(card)
    
    return html.Div(stat_cards)

def create_volatility_tab(data, crypto):
    """Create volatility analysis tab"""
    volatility_forecast = data.get('volatility_forecast', {})
    
    if not volatility_forecast:
        return html.P("No volatility forecast available. Select GARCH or EGARCH model.")
    
    # Create volatility comparison chart
    fig = go.Figure()
    
    for model, forecast in volatility_forecast.items():
        x_values = list(range(1, len(forecast) + 1))
        fig.add_trace(go.Scatter(
            x=x_values,
            y=forecast,
            mode='lines+markers',
            name=model,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f"{crypto} Volatility Forecast Comparison",
        xaxis_title="Forecast Day",
        yaxis_title="Volatility (%)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    # AIC/BIC comparison
    metrics = data.get('model_metrics', {})
    vol_models = [m for m in metrics.keys() if m in ['GARCH', 'EGARCH']]
    
    if vol_models:
        aic_values = [metrics[m].get('aic', 0) for m in vol_models]
        bic_values = [metrics[m].get('bic', 0) for m in vol_models]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=vol_models, y=aic_values, name='AIC', marker_color=COLORS['warning']))
        fig2.add_trace(go.Bar(x=vol_models, y=bic_values, name='BIC', marker_color=COLORS['success']))
        
        fig2.update_layout(
            title="AIC/BIC Comparison (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="Information Criterion",
            template="plotly_white",
            height=300,
            barmode='group'
        )
    else:
        fig2 = None
    
    content = [
        dcc.Graph(figure=fig),
        html.Hr(style={'margin': '20px 0'})
    ]
    
    if fig2:
        content.extend([
            html.H4("Model Selection Criteria", style={'color': COLORS['primary']}),
            dcc.Graph(figure=fig2)
        ])
    
    return html.Div(content)

def create_comparison_tab(data):
    """Create model comparison tab"""
    metrics = data.get('model_metrics', {})
    best_models = data.get('best_models', {})
    
    if not metrics:
        return html.P("No comparison data available")
    
    # Create ranking table
    price_models = {k: v for k, v in metrics.items() if k in ['ARIMA', 'LSTM']}
    vol_models = {k: v for k, v in metrics.items() if k in ['GARCH', 'EGARCH']}
    
    content = []
    
    # Price models ranking
    if price_models:
        price_ranking = sorted(price_models.items(), key=lambda x: x[1].get('rmse', float('inf')))
        
        content.append(html.Div([
            html.H4("💰 Price Forecasting Models Ranking", style={'color': COLORS['primary']}),
            html.Div([
                html.Div([
                    html.Div(f"#{i+1}", style={
                        'fontSize': '24px', 
                        'fontWeight': 'bold',
                        'color': COLORS['success'] if i == 0 else COLORS['text_light'],
                        'marginRight': '15px'
                    }),
                    html.Div([
                        html.Div(model, style={'fontSize': '18px', 'fontWeight': 'bold'}),
                        html.Div(f"RMSE: {metrics_data.get('rmse', 0):.2f} | MAE: {metrics_data.get('mae', 0):.2f}",
                                style={'fontSize': '14px', 'color': COLORS['text_light']})
                    ])
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'padding': '15px',
                    'backgroundColor': COLORS['background'] if i == 0 else 'white',
                    'borderRadius': '8px',
                    'marginBottom': '10px',
                    'border': f'2px solid {COLORS["success"]}' if i == 0 else f'1px solid {COLORS["border"]}'
                }) for i, (model, metrics_data) in enumerate(price_ranking)
            ])
        ], style={'marginBottom': '30px'}))
    
    # Volatility models ranking
    if vol_models:
        vol_ranking = sorted(vol_models.items(), key=lambda x: x[1].get('aic', float('inf')))
        
        content.append(html.Div([
            html.H4("📊 Volatility Forecasting Models Ranking", style={'color': COLORS['primary']}),
            html.Div([
                html.Div([
                    html.Div(f"#{i+1}", style={
                        'fontSize': '24px',
                        'fontWeight': 'bold',
                        'color': COLORS['warning'] if i == 0 else COLORS['text_light'],
                        'marginRight': '15px'
                    }),
                    html.Div([
                        html.Div(model, style={'fontSize': '18px', 'fontWeight': 'bold'}),
                        html.Div(f"AIC: {metrics_data.get('aic', 0):.2f} | BIC: {metrics_data.get('bic', 0):.2f}",
                                style={'fontSize': '14px', 'color': COLORS['text_light']})
                    ])
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'padding': '15px',
                    'backgroundColor': COLORS['background'] if i == 0 else 'white',
                    'borderRadius': '8px',
                    'marginBottom': '10px',
                    'border': f'2px solid {COLORS["warning"]}' if i == 0 else f'1px solid {COLORS["border"]}'
                }) for i, (model, metrics_data) in enumerate(vol_ranking)
            ])
        ]))
    
    # Best models summary
    if best_models:
        content.append(html.Div([
            html.H4("🏆 Recommended Models", style={'color': COLORS['primary'], 'marginTop': '30px'}),
            html.Div([
                html.Div([
                    html.Span("Best for Price Forecasting: ", style={'fontWeight': 'bold'}),
                    html.Span(best_models.get('price', 'N/A'), 
                             style={'color': COLORS['success'], 'fontSize': '20px', 'fontWeight': 'bold'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Span("Best for Volatility Modeling: ", style={'fontWeight': 'bold'}),
                    html.Span(best_models.get('volatility', 'N/A'),
                             style={'color': COLORS['warning'], 'fontSize': '20px', 'fontWeight': 'bold'})
                ])
            ], style={
                'padding': '20px',
                'backgroundColor': COLORS['background'],
                'borderRadius': '8px',
                'marginTop': '15px'
            })
        ]))
    
    return html.Div(content)

def run_dashboard(debug=True, port=8050):
    """Run the dashboard"""
    print("="*70)
    print("🔮 CRYPTOCURRENCY FORECASTING DASHBOARD")
    print("="*70)
    print(f"\n🌐 Dashboard URL: http://localhost:{port}")
    print("📊 Features:")
    print("   • Real-time forecasting with ARIMA, LSTM, GARCH, EGARCH")
    print("   • Interactive visualizations with confidence intervals")
    print("   • Comprehensive model performance metrics")
    print("   • Residual diagnostics and volatility analysis")
    print("   • Model comparison and ranking")
    print(f"\n⚠️  Make sure the API is running at {API_BASE_URL}")
    print("   Start API with: python run_api.py")
    print("\n" + "="*70 + "\n")
    
    app.run_server(debug=debug, port=port, host='0.0.0.0')

if __name__ == '__main__':
    run_dashboard(debug=True, port=8050)