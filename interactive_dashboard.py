#!/usr/bin/env python3
"""
Interactive Cryptocurrency Forecasting Dashboard
Week 9: Dashboard Development - Complete Implementation
Uses Dash/Plotly for real-time interactive visualization
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import json

# ── Color Palette (Deep Navy / Sapphire) ──────────────────────────────────────
C = {
    'bg':        '#060A14',
    'surface':   '#0C1525',
    'surface2':  '#112038',
    'surface3':  '#172B4A',
    'border':    '#1C3356',
    'border2':   '#244470',
    'accent':    '#3B82F6',   # Sapphire blue
    'accent2':   '#2563EB',
    'btc':       '#F5A623',   # Gold
    'btc_dim':   '#C98319',
    'eth':       '#818CF8',   # Soft indigo
    'success':   '#10B981',   # Emerald
    'warning':   '#F59E0B',   # Amber
    'danger':    '#EF4444',   # Rose
    'text':      '#F0F6FF',
    'text2':     '#8BA3C4',
    'text3':     '#445577',
}

API_BASE_URL = "http://localhost:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #060A14;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F0F6FF;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0C1525; }
::-webkit-scrollbar-thumb { background: #1C3356; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3B82F6; }

/* ── Pulse animations ── */
@keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70%  { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70%  { box-shadow: 0 0 0 9px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Status dot ── */
.status-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #10B981;
    display: inline-block;
    animation: pulse-green 2.2s infinite;
    margin-right: 8px;
    vertical-align: middle;
    flex-shrink: 0;
}
.status-dot.error {
    background: #EF4444;
    animation: pulse-red 2.2s infinite;
}

/* ── Generate button ── */
#generate-button {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 13px 28px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.40), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    letter-spacing: 0.4px !important;
    white-space: nowrap !important;
}
#generate-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(59,130,246,0.60) !important;
}
#generate-button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.3) !important;
}

/* ── Dropdown overrides ── */
.VirtualizedSelectFocusedOption,
.Select-option.is-focused { background: #172B4A !important; }
.Select-option { background: #112038 !important; color: #F0F6FF !important; }
.Select-option:hover { background: #172B4A !important; }
.Select-control {
    background: #112038 !important;
    border: 1px solid #1C3356 !important;
    border-radius: 10px !important;
    color: #F0F6FF !important;
    box-shadow: none !important;
}
.Select-control:hover { border-color: #3B82F6 !important; }
.Select-menu-outer {
    background: #112038 !important;
    border: 1px solid #1C3356 !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
    overflow: hidden !important;
}
.Select-value-label { color: #F0F6FF !important; }
.Select-placeholder { color: #8BA3C4 !important; }
.Select-arrow { border-top-color: #8BA3C4 !important; }
.Select-input > input { color: #F0F6FF !important; }
.is-open .Select-arrow { border-bottom-color: #8BA3C4 !important; }

/* ── Slider track/handle ── */
.rc-slider-rail { background: #1C3356 !important; height: 4px !important; }
.rc-slider-track { background: linear-gradient(90deg, #3B82F6, #10B981) !important; height: 4px !important; }
.rc-slider-handle {
    border: 2px solid #3B82F6 !important;
    background: #0C1525 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
    width: 16px !important; height: 16px !important;
    margin-top: -6px !important;
}
.rc-slider-handle:hover { box-shadow: 0 0 0 5px rgba(59,130,246,0.35) !important; }
.rc-slider-tooltip-inner {
    background: #112038 !important;
    border: 1px solid #1C3356 !important;
    color: #F0F6FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    border-radius: 6px !important;
}
.rc-slider-mark-text { color: #445577 !important; font-size: 11px !important; }

/* ── Tab styling ── */
.custom-tab {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #8BA3C4 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 14px 22px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.custom-tab:hover { color: #F0F6FF !important; }
.custom-tab--selected {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
    background: transparent !important;
}

/* ── Loading ── */
._dash-loading-callback { opacity: 0.6; }
.dash-loading > * { color: #3B82F6 !important; }

/* ── Fancy table ── */
.fancy-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.fancy-table thead th {
    background: #112038;
    color: #8BA3C4;
    font-weight: 600;
    padding: 11px 16px;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    border-bottom: 1px solid #1C3356;
    font-family: 'Inter', sans-serif;
}
.fancy-table tbody tr { border-bottom: 1px solid #112038; transition: background 0.15s; }
.fancy-table tbody tr:hover { background: #112038; }
.fancy-table tbody td {
    padding: 11px 16px;
    color: #F0F6FF;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Metric card hover ── */
.metric-card { transition: transform 0.2s ease, box-shadow 0.2s ease; }
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* ── Ranking row hover ── */
.rank-row { transition: transform 0.15s ease; }
.rank-row:hover { transform: translateX(3px); }

/* ── Gradient text utility ── */
.grad-text-btc {
    background: linear-gradient(135deg, #F7931A, #FFD166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.grad-text-accent {
    background: linear-gradient(135deg, #7C6FFF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Card animation ── */
.animated-card { animation: fadeInUp 0.4s ease both; }
"""

# ── App init ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="CryptoForecast",
    update_title="Generating...",
    suppress_callback_exceptions=True
)

app.index_string = f'''<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{CUSTOM_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>'''


# ── Style helpers ──────────────────────────────────────────────────────────────
def card(extra=None):
    base = {
        'background': C['surface'],
        'border': f'1px solid {C["border"]}',
        'borderRadius': '16px',
        'padding': '24px',
        'boxShadow': '0 2px 16px rgba(0,0,0,0.35)',
    }
    if extra:
        base.update(extra)
    return base


def label_style():
    return {
        'color': C['text2'],
        'fontSize': '11px',
        'fontWeight': '600',
        'textTransform': 'uppercase',
        'letterSpacing': '0.9px',
        'marginBottom': '8px',
        'display': 'block',
        'fontFamily': "'Inter', sans-serif"
    }


def section_title(text, icon='', color=None):
    return html.Div([
        html.Span(icon + ' ' if icon else '', style={'marginRight': '6px'}),
        html.Span(text, style={
            'color': color or C['text'],
            'fontWeight': '700',
            'fontSize': '15px',
            'fontFamily': "'Space Grotesk', sans-serif",
            'letterSpacing': '0.2px'
        })
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '18px'})


# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── Top Nav Bar ───────────────────────────────────────────────────────────
    html.Div([
        # Brand
        html.Div([
            html.Div([
                html.Span("◈", style={
                    'fontSize': '22px', 'marginRight': '10px',
                    'background': 'linear-gradient(135deg, #F7931A 0%, #7C6FFF 100%)',
                    'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
                    'fontWeight': '800'
                }),
                html.Span("CRYPTO", style={
                    'fontSize': '17px', 'fontWeight': '800', 'letterSpacing': '3px',
                    'color': C['text'], 'fontFamily': "'Space Grotesk', sans-serif"
                }),
                html.Span("FORECAST", style={
                    'fontSize': '17px', 'fontWeight': '800', 'letterSpacing': '3px',
                    'background': 'linear-gradient(90deg, #7C6FFF, #00D4AA)',
                    'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
                    'fontFamily': "'Space Grotesk', sans-serif", 'marginLeft': '4px'
                })
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.P("Advanced Time Series · Volatility Modeling", style={
                'color': C['text3'], 'fontSize': '12px', 'marginTop': '3px',
                'letterSpacing': '0.4px'
            })
        ], style={'flex': '1'}),

        # Status
        html.Div([
            html.Div([
                html.Span(id='status-dot', className='status-dot'),
                html.Span("API Connected", id='api-status', style={
                    'color': C['success'], 'fontWeight': '600', 'fontSize': '13px'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end'}),
            html.Div(id='last-update', style={
                'color': C['text3'], 'fontSize': '11px',
                'marginTop': '3px', 'textAlign': 'right'
            })
        ])
    ], style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '16px 32px',
        'background': C['surface'],
        'borderBottom': f'1px solid {C["border"]}',
        'position': 'sticky', 'top': '0', 'zIndex': '200',
    }),

    # ── Page content ──────────────────────────────────────────────────────────
    html.Div([

        # ── Control Panel ─────────────────────────────────────────────────────
        html.Div([

            # Crypto select
            html.Div([
                html.Span("Cryptocurrency", style=label_style()),
                dcc.Dropdown(
                    id='crypto-dropdown',
                    options=[
                        {'label': '₿  Bitcoin (BTC)', 'value': 'BTC'},
                        {'label': 'Ξ  Ethereum (ETH)', 'value': 'ETH'}
                    ],
                    value='BTC', clearable=False,
                    style={'width': '195px'}
                )
            ], style={'marginRight': '20px', 'flexShrink': '0'}),

            # Divider
            html.Div(style={
                'width': '1px', 'background': C['border'],
                'alignSelf': 'stretch', 'margin': '0 4px 0 0', 'flexShrink': '0'
            }),

            # Model select
            html.Div([
                html.Span("Model", style=label_style()),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': '📈  ARIMA (Price)', 'value': 'ARIMA'},
                        {'label': '🤖  LSTM (Price)', 'value': 'LSTM'},
                        {'label': '📊  GARCH (Volatility)', 'value': 'GARCH'},
                        {'label': '⚡  EGARCH (Volatility)', 'value': 'EGARCH'}
                    ],
                    value='ARIMA', clearable=False,
                    style={'width': '215px'}
                )
            ], style={'marginRight': '20px', 'flexShrink': '0'}),

            html.Div(style={
                'width': '1px', 'background': C['border'],
                'alignSelf': 'stretch', 'margin': '0 8px 0 0', 'flexShrink': '0'
            }),

            # Forecast slider
            html.Div([
                html.Span("Forecast Horizon", style=label_style()),
                dcc.Slider(
                    id='forecast-slider', min=3, max=30, step=1, value=7,
                    marks={3: {'label': '3d', 'style': {'color': C['text3']}},
                           7: {'label': '7d', 'style': {'color': C['text3']}},
                           14: {'label': '14d', 'style': {'color': C['text3']}},
                           21: {'label': '21d', 'style': {'color': C['text3']}},
                           30: {'label': '30d', 'style': {'color': C['text3']}}},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'flex': '1', 'marginRight': '24px'}),

            # Confidence slider
            html.Div([
                html.Span("Confidence Level", style=label_style()),
                dcc.Slider(
                    id='confidence-slider', min=80, max=99, step=1, value=95,
                    marks={80: {'label': '80%', 'style': {'color': C['text3']}},
                           90: {'label': '90%', 'style': {'color': C['text3']}},
                           95: {'label': '95%', 'style': {'color': C['text3']}},
                           99: {'label': '99%', 'style': {'color': C['text3']}}},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '240px', 'marginRight': '24px', 'flexShrink': '0'}),

            # CTA
            html.Button('⚡  Generate Forecast', id='generate-button', n_clicks=0,
                        style={'alignSelf': 'flex-end', 'flexShrink': '0'})

        ], style={
            **card(),
            'display': 'flex', 'alignItems': 'flex-end',
            'gap': '0', 'marginBottom': '20px',
            'padding': '20px 24px',
            'borderTop': f'2px solid {C["accent2"]}',
        }),

        # ── Chart + Sidebar row ───────────────────────────────────────────────
        html.Div([

            # Main forecast chart
            html.Div([
                dcc.Loading(
                    id='loading-chart', type='circle', color=C['accent'],
                    children=[dcc.Graph(
                        id='main-forecast-chart',
                        style={'height': '460px'},
                        config={'displayModeBar': True, 'displaylogo': False,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d']}
                    )]
                )
            ], style={**card(), 'flex': '2', 'marginRight': '20px', 'padding': '16px',
                      'borderTop': f'2px solid {C["btc"]}'}),

            # Metrics sidebar
            html.Div([
                section_title("Model Performance", "◎", C['accent']),
                html.Div(id='metrics-panel'),

                # Divider
                html.Div(style={
                    'height': '1px', 'margin': '20px 0',
                    'background': f'linear-gradient(90deg, {C["accent"]}60, transparent)'
                }),

                section_title("Best Models", "🏆"),
                html.Div(id='best-models-panel')

            ], style={
                **card(),
                'flex': '1', 'maxHeight': '496px',
                'overflowY': 'auto', 'padding': '22px',
                'borderTop': f'2px solid {C["success"]}'
            })

        ], style={'display': 'flex', 'marginBottom': '20px'}),

        # ── Analysis Tabs ─────────────────────────────────────────────────────
        html.Div([
            dcc.Tabs(
                id='analysis-tabs', value='performance',
                children=[
                    dcc.Tab(label='📈  Performance',  value='performance',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='🔬  Residuals',    value='residuals',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='⚡  Volatility',   value='volatility',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='🔄  Comparison',   value='comparison',
                            className='custom-tab', selected_className='custom-tab--selected'),
                ],
                style={'borderBottom': f'1px solid {C["border"]}'},
                colors={'border': 'transparent', 'primary': C['accent'],
                        'background': C['surface']}
            ),
            html.Div(id='tab-content', style={'padding': '24px'})
        ], style={**card(), 'padding': '0', 'marginBottom': '20px',
                  'overflow': 'hidden'}),

        # ── Footer ────────────────────────────────────────────────────────────
        html.Div([
            html.Span("◈ ", style={'color': C['accent']}),
            html.Span("CryptoForecast", style={
                'color': C['text2'], 'fontWeight': '700',
                'fontFamily': "'Space Grotesk', sans-serif"
            }),
            html.Span("  ·  ARIMA  ·  LSTM  ·  GARCH  ·  EGARCH",
                      style={'color': C['text3'], 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'padding': '20px',
                  'borderTop': f'1px solid {C["border"]}'}),

    ], style={'padding': '24px', 'maxWidth': '1700px', 'margin': '0 auto'}),

    # Data store
    html.Div(id='forecast-data-store', style={'display': 'none'})

], style={'backgroundColor': C['bg'], 'minHeight': '100vh'})


# ── Helpers ────────────────────────────────────────────────────────────────────

def empty_fig(msg="Select parameters above and click  ⚡ Generate Forecast"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=C['surface'], plot_bgcolor=C['surface'],
        height=460,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(
            text=msg, xref='paper', yref='paper',
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=15, color=C['text2'], family='Inter'),
            showarrow=False
        )]
    )
    return fig


def _dark_base(fig, title, xlab, ylab, height=380):
    fig.update_layout(
        paper_bgcolor=C['surface'], plot_bgcolor=C['surface'],
        height=height,
        font=dict(family='Inter, sans-serif', color=C['text2']),
        title=dict(text=f'<b>{title}</b>',
                   font=dict(size=14, color=C['text'],
                             family='Space Grotesk, sans-serif'), x=0.01),
        xaxis=dict(title=dict(text=xlab, font=dict(size=12, color=C['text2'])),
                   gridcolor=C['border'], zeroline=False,
                   tickfont=dict(color=C['text2']), showline=False),
        yaxis=dict(title=dict(text=ylab, font=dict(size=12, color=C['text2'])),
                   gridcolor=C['border'], zeroline=False,
                   tickfont=dict(color=C['text2']), showline=False),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=C['surface2'], bordercolor=C['border'],
                        font=dict(color=C['text'], size=12, family='Inter')),
        legend=dict(font=dict(color=C['text2'], size=12),
                    bgcolor='rgba(0,0,0,0)', bordercolor='transparent'),
        margin=dict(l=60, r=24, t=52, b=50),
    )
    return fig


def _no_data():
    return html.Div("Generate a forecast to see results", style={
        'color': C['text3'], 'fontSize': '13px',
        'textAlign': 'center', 'padding': '24px'
    })


def _err(msg):
    return html.Div(f"⚠  {msg}", style={
        'color': C['danger'], 'fontSize': '13px',
        'padding': '8px 12px', 'background': f'{C["danger"]}18',
        'borderRadius': '8px', 'border': f'1px solid {C["danger"]}40'
    })


# ── Forecast callback ──────────────────────────────────────────────────────────

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
    ok_style = {'color': C['success'], 'fontWeight': '600', 'fontSize': '13px'}
    err_style = {'color': C['danger'], 'fontWeight': '600', 'fontSize': '13px'}

    if n_clicks == 0:
        return (empty_fig(), _no_data(), _no_data(),
                json.dumps({}), "", "API Connected", ok_style)

    try:
        resp = requests.post(
            f"{API_BASE_URL}/forecast",
            json={"cryptocurrency": crypto, "forecast_days": forecast_days,
                  "confidence_level": confidence / 100, "models": [model]},
            timeout=60
        )
        if resp.status_code != 200:
            raise Exception(f"API returned {resp.status_code}")

        data = resp.json()
        fig = _build_forecast_chart(data, model, crypto, confidence)
        metrics_el = _build_metrics_panel(data, model)
        best_el = _build_best_models_panel(data)
        ts = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return (fig, metrics_el, best_el, json.dumps(data), ts,
                "API Connected", ok_style)

    except Exception as e:
        return (empty_fig(f"⚠  {e}"),
                _err(str(e)), _err(str(e)),
                json.dumps({}),
                f"Error at {datetime.now().strftime('%H:%M:%S')}",
                "API Error", err_style)


# ── Chart builder ──────────────────────────────────────────────────────────────

def _build_forecast_chart(data, model, crypto, confidence):
    color = C['btc'] if crypto == 'BTC' else C['eth']
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    if model in ['ARIMA', 'LSTM']:
        forecast = data['price_forecast'].get(model, [])
        ci = data['price_confidence_intervals'].get(model, [])
        y_title = "Price (USD)"
        subtitle = f"{crypto} Price Forecast"
    else:
        forecast = data['volatility_forecast'].get(model, [])
        ci = []
        y_title = "Volatility (%)"
        subtitle = f"{crypto} Volatility Forecast"

    x = list(range(1, len(forecast) + 1))
    fig = go.Figure()

    # CI fill
    if ci:
        upper = [c['upper'] for c in ci]
        lower = [c['lower'] for c in ci]
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=upper + lower[::-1],
            fill='toself',
            fillcolor=f'rgba({r},{g},{b},0.10)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{confidence}% Confidence', hoverinfo='skip', showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=x, y=upper, mode='lines',
            line=dict(color=f'rgba({r},{g},{b},0.35)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=lower, mode='lines',
            line=dict(color=f'rgba({r},{g},{b},0.35)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=x, y=forecast,
        mode='lines+markers',
        name=f'{model} Forecast',
        line=dict(color=color, width=2.5, shape='spline'),
        marker=dict(size=7, color=color,
                    symbol='circle',
                    line=dict(color=C['surface'], width=2)),
        hovertemplate=f'<b>Day %{{x}}</b><br>{y_title}: <b>%{{y:,.2f}}</b><extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor=C['surface'], plot_bgcolor=C['surface'],
        height=460,
        font=dict(family='Inter, sans-serif', color=C['text2']),
        title=dict(
            text=(f'<b style="color:{C["text"]};font-family:Space Grotesk">{subtitle}</b>'
                  f'<span style="font-size:13px;color:{C["text2"]}">  —  {model} Model</span>'),
            font=dict(size=17, family='Space Grotesk, sans-serif'),
            x=0.02, y=0.96
        ),
        xaxis=dict(
            title=dict(text='Forecast Day', font=dict(size=12, color=C['text2'])),
            gridcolor=C['border'], zeroline=False, showline=False,
            tickfont=dict(color=C['text2']), tickcolor=C['border'],
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=C['text2'])),
            gridcolor=C['border'], zeroline=False, showline=False,
            tickfont=dict(color=C['text2']), tickcolor=C['border'],
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=C['surface2'], bordercolor=color,
                        font=dict(color=C['text'], size=13, family='Inter')),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='right', x=1,
            font=dict(color=C['text2'], size=12),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=20, t=68, b=50),
    )
    return fig


# ── Metrics panel ──────────────────────────────────────────────────────────────

def _build_metrics_panel(data, model):
    metrics = data['model_metrics'].get(model, {})
    if not metrics:
        return _no_data()

    defs = ([('RMSE', 'rmse', C['warning'], '±'),
             ('MAE',  'mae',  C['success'], '↔'),
             ('MAPE', 'mape', C['danger'],  '%')]
            if model in ['ARIMA', 'LSTM'] else
            [('AIC',  'aic',  C['warning'], '↓'),
             ('BIC',  'bic',  C['success'], '↓'),
             ('RMSE', 'rmse', C['danger'],  '±')])

    cards = []
    for label, key, color, icon in defs:
        if key not in metrics:
            continue
        val = metrics[key]
        display = f"{val:.2f}{'%' if key == 'mape' else ''}"
        cards.append(html.Div([
            html.Div([
                html.Span(icon, style={
                    'color': color, 'fontSize': '15px', 'fontWeight': '700',
                    'marginRight': '7px', 'opacity': '0.75'
                }),
                html.Span(label, style={
                    'color': C['text2'], 'fontSize': '11px', 'fontWeight': '600',
                    'textTransform': 'uppercase', 'letterSpacing': '0.8px'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '7px'}),
            html.Div(display, style={
                'fontSize': '24px', 'fontWeight': '700', 'color': color,
                'fontFamily': "'Space Grotesk', sans-serif",
                'textShadow': f'0 0 24px {color}40'
            })
        ], className='metric-card', style={
            'padding': '14px 16px',
            'background': f'linear-gradient(135deg, {color}0D, {C["surface2"]})',
            'borderRadius': '12px',
            'marginBottom': '10px',
            'borderLeft': f'3px solid {color}',
            'border': f'1px solid {color}25',
            'borderLeftWidth': '3px',
            'borderLeftColor': color
        }))

    return html.Div(cards)


def _build_best_models_panel(data):
    best = data.get('best_models', {})
    if not best:
        return _no_data()

    panels = []
    if 'price' in best:
        panels.append(_best_badge("Price", best['price'], C['btc'], "₿"))
    if 'volatility' in best:
        panels.append(_best_badge("Volatility", best['volatility'], C['accent'], "⚡"))
    return html.Div(panels)


def _best_badge(category, name, color, icon):
    return html.Div([
        html.Div(f"{icon}  Best {category} Model", style={
            'color': C['text2'], 'fontSize': '11px', 'fontWeight': '600',
            'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '8px'
        }),
        html.Div(name, style={
            'fontSize': '20px', 'fontWeight': '700', 'color': color,
            'fontFamily': "'Space Grotesk', sans-serif",
            'textShadow': f'0 0 20px {color}55',
            'padding': '10px 16px', 'textAlign': 'center',
            'background': f'linear-gradient(135deg, {color}18, {color}06)',
            'borderRadius': '10px',
            'border': f'1px solid {color}35',
            'letterSpacing': '1px'
        })
    ], style={'marginBottom': '14px'})


# ── Tab callback ───────────────────────────────────────────────────────────────

@app.callback(
    Output('tab-content', 'children'),
    [Input('analysis-tabs', 'value')],
    [State('forecast-data-store', 'children'),
     State('crypto-dropdown', 'value')]
)
def update_tab_content(tab, store, crypto):
    if not store or store == '{}':
        return html.Div("Generate a forecast above to see detailed analysis",
                        style={'textAlign': 'center', 'color': C['text2'],
                               'padding': '60px 0', 'fontSize': '14px'})
    try:
        data = json.loads(store)
    except Exception:
        return html.Div("Error loading data", style={'color': C['danger']})

    if tab == 'performance':
        return _tab_performance(data, crypto)
    elif tab == 'residuals':
        return _tab_residuals(data)
    elif tab == 'volatility':
        return _tab_volatility(data, crypto)
    elif tab == 'comparison':
        return _tab_comparison(data)
    return html.Div()


# ── Tab: Performance ───────────────────────────────────────────────────────────

def _tab_performance(data, crypto):
    metrics = data.get('model_metrics', {})
    if not metrics:
        return html.Div("No metrics available", style={'color': C['text2']})

    models = list(metrics.keys())
    rmse_vals = [metrics[m].get('rmse', 0) for m in models]
    mae_vals = [metrics[m].get('mae', 0) for m in models]
    bar_colors = [C['btc'], C['accent'], C['success'], C['eth']]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE by Model', 'MAE by Model'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    fig.add_trace(go.Bar(
        x=models, y=rmse_vals, name='RMSE',
        marker=dict(color=bar_colors[:len(models)]),
        hovertemplate='%{x}: <b>%{y:.2f}</b><extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=models, y=mae_vals, name='MAE',
        marker=dict(color=[C['success'], C['warning'], C['eth'], C['danger']][:len(models)]),
        hovertemplate='%{x}: <b>%{y:.2f}</b><extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=C['surface'], plot_bgcolor=C['surface'],
        height=340, showlegend=False,
        font=dict(family='Inter, sans-serif', color=C['text2']),
        title=dict(text=f'<b>{crypto} Model Performance</b>',
                   font=dict(size=14, color=C['text'], family='Space Grotesk'), x=0.01),
        hoverlabel=dict(bgcolor=C['surface2'], bordercolor=C['border'],
                        font=dict(color=C['text'], size=12)),
        margin=dict(l=50, r=24, t=50, b=50),
    )
    for i in range(1, 3):
        fig.update_xaxes(gridcolor=C['border'], zeroline=False,
                         tickfont=dict(color=C['text2']), showline=False, row=1, col=i)
        fig.update_yaxes(gridcolor=C['border'], zeroline=False,
                         tickfont=dict(color=C['text2']), showline=False, row=1, col=i)
    for ann in fig.layout.annotations:
        ann.font.color = C['text2']
        ann.font.size = 12

    # Metrics table
    num_keys = sorted({k for md in metrics.values()
                       for k, v in md.items() if isinstance(v, (int, float))})
    head_cells = [html.Th('Model', style={
        'padding': '11px 16px', 'background': C['surface2'],
        'color': C['text2'], 'fontSize': '11px',
        'textTransform': 'uppercase', 'letterSpacing': '0.9px',
        'fontWeight': '600', 'textAlign': 'left'
    })]
    for k in num_keys:
        head_cells.append(html.Th(k.upper(), style={
            'padding': '11px 16px', 'background': C['surface2'],
            'color': C['text2'], 'fontSize': '11px',
            'textTransform': 'uppercase', 'letterSpacing': '0.9px',
            'fontWeight': '600', 'textAlign': 'left'
        }))

    body_rows = []
    for m, md in metrics.items():
        cells = [html.Td(m, style={
            'padding': '11px 16px', 'fontWeight': '600',
            'color': C['text'], 'fontFamily': 'Space Grotesk, sans-serif'
        })]
        for k in num_keys:
            v = md.get(k, '—')
            cells.append(html.Td(
                f"{v:.2f}{'%' if k == 'mape' else ''}" if isinstance(v, (int, float)) else str(v),
                style={'padding': '11px 16px', 'color': C['text2'],
                       'fontFamily': 'Space Grotesk, sans-serif'}
            ))
        body_rows.append(html.Tr(cells, style={'borderBottom': f'1px solid {C["border"]}'}))

    return html.Div([
        dcc.Graph(figure=fig),
        html.Div(style={'height': '1px', 'background': C['border'], 'margin': '20px 0'}),
        html.Div("Detailed Metrics Table", style={
            'color': C['text2'], 'fontSize': '11px', 'fontWeight': '600',
            'textTransform': 'uppercase', 'letterSpacing': '0.9px', 'marginBottom': '12px'
        }),
        html.Div([
            html.Table([
                html.Thead(html.Tr(head_cells)),
                html.Tbody(body_rows)
            ], className='fancy-table')
        ], style={'borderRadius': '12px', 'overflow': 'hidden',
                  'border': f'1px solid {C["border"]}'})
    ])


# ── Tab: Residuals ─────────────────────────────────────────────────────────────

def _tab_residuals(data):
    stats = data.get('residual_statistics', {})
    if not stats:
        return html.Div("Residual analysis available for ARIMA and LSTM models",
                        style={'color': C['text2'], 'textAlign': 'center', 'padding': '50px'})

    cards = []
    for model, s in stats.items():
        items = [
            ('Mean',     s['mean'],     '≈ 0 ideal', abs(s['mean']) < 0.5),
            ('Std Dev',  s['std'],      '',           True),
            ('Skewness', s['skewness'], '≈ 0 ideal', abs(s['skewness']) < 0.5),
            ('Kurtosis', s['kurtosis'], '≈ 3 ideal', abs(s['kurtosis'] - 3) < 1),
        ]
        stat_divs = []
        for sname, sval, hint, ok in items:
            c = C['success'] if ok else C['warning']
            stat_divs.append(html.Div([
                html.Div(sname, style={
                    'color': C['text2'], 'fontSize': '11px', 'fontWeight': '600',
                    'textTransform': 'uppercase', 'letterSpacing': '0.8px'
                }),
                html.Div(f"{sval:.4f}", style={
                    'fontSize': '22px', 'fontWeight': '700', 'color': c,
                    'fontFamily': "'Space Grotesk', sans-serif", 'margin': '4px 0',
                    'textShadow': f'0 0 16px {c}40'
                }),
                html.Div(hint, style={'color': C['text3'], 'fontSize': '11px'})
            ], style={
                'flex': '1', 'padding': '16px', 'textAlign': 'center',
                'background': C['surface2'], 'borderRadius': '12px',
                'borderBottom': f'2px solid {c}'
            }))

        cards.append(html.Div([
            html.Div(model, style={
                'color': C['text'], 'fontWeight': '700', 'fontSize': '16px',
                'fontFamily': "'Space Grotesk', sans-serif", 'marginBottom': '16px'
            }),
            html.Div(stat_divs, style={'display': 'flex', 'gap': '10px'})
        ], style={
            **card(), 'marginBottom': '16px',
            'borderTop': f'2px solid {C["accent"]}'
        }))

    return html.Div(cards)


# ── Tab: Volatility ────────────────────────────────────────────────────────────

def _tab_volatility(data, crypto):
    vol = data.get('volatility_forecast', {})
    if not vol:
        return html.Div("Select GARCH or EGARCH and generate forecast",
                        style={'color': C['text2'], 'textAlign': 'center', 'padding': '50px'})

    palette = [C['accent'], C['success'], C['btc'], C['eth']]
    fig = go.Figure()
    for i, (model, forecast) in enumerate(vol.items()):
        x = list(range(1, len(forecast) + 1))
        col = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=x, y=forecast, mode='lines+markers', name=model,
            line=dict(color=col, width=2.5, shape='spline'),
            marker=dict(size=7, color=col, line=dict(color=C['surface'], width=2)),
            hovertemplate='<b>Day %{x}</b><br>Volatility: <b>%{y:.4f}</b><extra></extra>'
        ))
    _dark_base(fig, f'{crypto} Volatility Forecast', 'Forecast Day', 'Volatility (%)')
    content = [dcc.Graph(figure=fig)]

    metrics = data.get('model_metrics', {})
    vm = [m for m in metrics if m in ['GARCH', 'EGARCH']]
    if vm:
        aic_vals = [metrics[m].get('aic', 0) for m in vm]
        bic_vals = [metrics[m].get('bic', 0) for m in vm]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=vm, y=aic_vals, name='AIC',
                              marker=dict(color=C['warning'])))
        fig2.add_trace(go.Bar(x=vm, y=bic_vals, name='BIC',
                              marker=dict(color=C['accent'])))
        _dark_base(fig2, 'AIC / BIC — Lower is Better',
                   'Model', 'Information Criterion', height=300)
        fig2.update_layout(barmode='group')
        content += [
            html.Div(style={'height': '1px', 'background': C['border'], 'margin': '8px 0'}),
            dcc.Graph(figure=fig2)
        ]

    return html.Div(content)


# ── Tab: Comparison ────────────────────────────────────────────────────────────

def _tab_comparison(data):
    metrics = data.get('model_metrics', {})
    best = data.get('best_models', {})
    if not metrics:
        return html.Div("No comparison data", style={'color': C['text2']})

    price_m = {k: v for k, v in metrics.items() if k in ['ARIMA', 'LSTM']}
    vol_m   = {k: v for k, v in metrics.items() if k in ['GARCH', 'EGARCH']}
    content = []

    def rank_block(title, ranked, fmt_fn, accent):
        rows = []
        for i, (m, md) in enumerate(ranked):
            is_top = i == 0
            col = accent if is_top else C['text2']
            rows.append(html.Div([
                html.Div(f"#{i+1}", style={
                    'fontSize': '26px', 'fontWeight': '800', 'color': col,
                    'fontFamily': "'Space Grotesk', sans-serif",
                    'minWidth': '48px', 'textAlign': 'center',
                    'textShadow': f'0 0 14px {col}55' if is_top else 'none'
                }),
                html.Div([
                    html.Div(m, style={
                        'fontSize': '16px', 'fontWeight': '700', 'color': C['text'],
                        'fontFamily': "'Space Grotesk', sans-serif"
                    }),
                    html.Div(fmt_fn(md), style={
                        'fontSize': '12px', 'color': C['text2'], 'marginTop': '3px'
                    })
                ], style={'flex': '1'}),
                html.Div("★ Best" if is_top else "", style={
                    'color': accent, 'fontSize': '12px', 'fontWeight': '700',
                    'background': f'{accent}18', 'padding': '4px 10px',
                    'borderRadius': '20px', 'border': f'1px solid {accent}35'
                })
            ], className='rank-row', style={
                'display': 'flex', 'alignItems': 'center', 'gap': '16px',
                'padding': '14px 18px',
                'background': f'linear-gradient(135deg, {accent}0D, {C["surface2"]})' if is_top else C['surface2'],
                'borderRadius': '12px',
                'border': f'1px solid {accent}35' if is_top else f'1px solid {C["border"]}',
                'marginBottom': '10px'
            }))
        return html.Div([
            html.Div(title, style={
                'color': C['text2'], 'fontSize': '11px', 'fontWeight': '600',
                'textTransform': 'uppercase', 'letterSpacing': '0.9px',
                'marginBottom': '14px'
            }),
            html.Div(rows)
        ], style={**card(), 'marginBottom': '16px', 'borderTop': f'2px solid {accent}'})

    if price_m:
        ranked = sorted(price_m.items(), key=lambda x: x[1].get('rmse', float('inf')))
        content.append(rank_block(
            "Price Forecasting — Ranked by RMSE", ranked,
            lambda md: f"RMSE: {md.get('rmse',0):.2f}  ·  MAE: {md.get('mae',0):.2f}  ·  MAPE: {md.get('mape',0):.2f}%",
            C['btc']
        ))

    if vol_m:
        ranked = sorted(vol_m.items(), key=lambda x: x[1].get('aic', float('inf')))
        content.append(rank_block(
            "Volatility Modeling — Ranked by AIC", ranked,
            lambda md: f"AIC: {md.get('aic',0):.2f}  ·  BIC: {md.get('bic',0):.2f}",
            C['accent']
        ))

    if best:
        content.append(html.Div([
            html.Div("Recommendations", style={
                'color': C['text2'], 'fontSize': '11px', 'fontWeight': '600',
                'textTransform': 'uppercase', 'letterSpacing': '0.9px',
                'marginBottom': '16px'
            }),
            html.Div([
                html.Div([
                    html.Span("₿  Best Price Model  ", style={'color': C['text2'], 'fontSize': '13px'}),
                    html.Span(best.get('price', 'N/A'), style={
                        'color': C['btc'], 'fontSize': '20px', 'fontWeight': '700',
                        'fontFamily': "'Space Grotesk', sans-serif",
                        'textShadow': f'0 0 16px {C["btc"]}55'
                    })
                ], style={'marginBottom': '14px'}),
                html.Div([
                    html.Span("⚡  Best Volatility Model  ", style={'color': C['text2'], 'fontSize': '13px'}),
                    html.Span(best.get('volatility', 'N/A'), style={
                        'color': C['accent'], 'fontSize': '20px', 'fontWeight': '700',
                        'fontFamily': "'Space Grotesk', sans-serif",
                        'textShadow': f'0 0 16px {C["accent"]}55'
                    })
                ])
            ], style={
                'padding': '20px 24px',
                'background': f'linear-gradient(135deg, {C["accent"]}10, {C["btc"]}08)',
                'borderRadius': '12px',
                'border': f'1px solid {C["border"]}'
            })
        ], style={**card()}))

    return html.Div(content)


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_dashboard(debug=True, port=8050):
    print("=" * 60)
    print("  ◈  CRYPTOFORECAST  —  DASHBOARD")
    print("=" * 60)
    print(f"  URL : http://localhost:{port}")
    print(f"  API : {API_BASE_URL}")
    print("=" * 60 + "\n")
    app.run(debug=debug, port=port, host='0.0.0.0')


if __name__ == '__main__':
    run_dashboard(debug=True, port=8050)
