import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def create_layout(df_reduced, service_cols, fig_corr):
    dashboard_style = {
        'height': '100vh',
        'overflow': 'hidden',
        'background-color': '#f8f9fa',
        'padding': '15px'
    }
    
    section_style = {
        'border': '2px solid #dee2e6',
        'border-radius': '10px',
        'padding': '15px',
        'margin-bottom': '15px',
        'box-shadow': '0 2px 5px 0 rgba(0,0,0,0.1)',
        'background-color': 'white',
        'height': '100%',
        'overflow': 'hidden'
    }

    header_style = {
        'color': '#2c3e50',
        'border-bottom': '1px solid #eee',
        'padding-bottom': '8px',
        'margin-bottom': '15px',
        'font-size': '1.2rem',
        'font-weight': 'bold'
    }

    return html.Div([
        # TITLE ROW
        html.Div([
            html.H1("✈️ Airline Satisfaction Dashboard", 
                   style={'text-align': 'center', 'color': '#3498db'}),
            html.P("Interactive dashboard for analyzing passenger metrics",
                  style={'text-align': 'center', 'color': '#7f8c8d'})
        ], style={'margin-bottom': '20px'}),
        
        # MAIN CONTENT ROWS
        dbc.Row([
            # FIRST ROW: DATA OVERVIEW | SERVICE RATING (40/60 split)
            dbc.Col([
                dbc.Row([
                    # DATA OVERVIEW (40%)
                    dbc.Col([
                        html.Div([
                            html.Div("DATA OVERVIEW", style=header_style),
                            dash_table.DataTable(
                                id='raw-data-table',
                                data=df_reduced.head(10).to_dict('records'),
                                columns=[{'name': col.replace('_', ' ').title(), 'id': col} 
                                       for col in df_reduced.columns],
                                style_table={'height': '300px', 'overflowY': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '6px',
                                    'minWidth': '80px',
                                    'maxWidth': '120px',
                                    'whiteSpace': 'normal'
                                },
                                style_header={
                                    'backgroundColor': '#3498db',
                                    'color': 'white',
                                    'fontWeight': 'bold'
                                },
                                page_size=10
                            )
                        ], style=section_style)
                    ], width=5),
                    
                    # SERVICE RATING (60%)
                    dbc.Col([
                        html.Div([
                            html.Div("SERVICE RATING ANALYSIS", style=header_style),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='service-dropdown',
                                        options=[{'label': col.replace('_', ' ').title(), 'value': col} 
                                               for col in service_cols],
                                        value='in_flight_entertainment',
                                        style={'margin-bottom': '10px'}
                                    ),
                                    dcc.Dropdown(
                                        id='chart-type',
                                        options=[
                                            {'label': 'Boxplot', 'value': 'box'},
                                            {'label': 'Grouped Bar Chart', 'value': 'bar'}
                                        ],
                                        value='box',
                                        style={'margin-bottom': '10px'}
                                    ),
                                    dash_table.DataTable(
                                        id='filtered-table',
                                        columns=[],
                                        data=[],
                                        style_table={'height': '150px', 'overflowY': 'auto'},
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '6px',
                                            'minWidth': '120px'
                                        },
                                        page_size=5
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    dcc.Graph(
                                        id='service-boxplot',
                                        style={'height': '300px'}
                                    )
                                ], width=6)
                            ])
                        ], style=section_style)
                    ], width=7)
                ]),
                
                # SECOND ROW: PREDICTIVE MODEL
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div("PREDICTIVE MODEL", style=header_style),
                            dbc.Tabs([
                                dbc.Tab(
                                    dbc.Card([
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Label("Model Type"),
                                                    dcc.Dropdown(
                                                        id='model-type',
                                                        options=[
                                                            {"label": "Logistic Regression", "value": "logreg"},
                                                            {"label": "Random Forest", "value": "rf"},
                                                            {"label": "XGBoost", "value": "xgb"},
                                                            {"label": "Neural Network", "value": "mlp"}
                                                        ],
                                                        value='rf',
                                                        clearable=False
                                                    )
                                                ], width=3),
                                                dbc.Col([
                                                    html.Label("Estimators/Iterations"),
                                                    dbc.Input(
                                                        id='n-estimators',
                                                        type='number',
                                                        value=100,
                                                        min=10,
                                                        max=1000,
                                                        step=10
                                                    )
                                                ], width=3),
                                                dbc.Col([
                                                    html.Label("Learning Rate"),
                                                    dbc.Input(
                                                        id='learning-rate',
                                                        type='number',
                                                        value=0.1,
                                                        min=0.001,
                                                        max=1,
                                                        step=0.01
                                                    )
                                                ], width=3),
                                                dbc.Col([
                                                    html.Label("Max Depth"),
                                                    dbc.Input(
                                                        id='max-depth',
                                                        type='number',
                                                        value=6,
                                                        min=1,
                                                        max=20,
                                                        step=1
                                                    )
                                                ], width=3)
                                            ], className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Label("Min Samples Split"),
                                                    dbc.Input(
                                                        id='min-samples-split',
                                                        type='number',
                                                        value=2,
                                                        min=2,
                                                        max=20,
                                                        step=1
                                                    )
                                                ], width=3),
                                                dbc.Col([
                                                    html.Label("Min Samples Leaf"),
                                                    dbc.Input(
                                                        id='min-samples-leaf',
                                                        type='number',
                                                        value=1,
                                                        min=1,
                                                        max=20,
                                                        step=1
                                                    )
                                                ], width=3),
                                                dbc.Col([
                                                    html.Label("Hidden Layer Sizes"),
                                                    dbc.Input(
                                                        id='hidden-layer-sizes',
                                                        type='text',
                                                        value="(100,)",
                                                        placeholder="Tuple format, e.g. (100,50)"
                                                    )
                                                ], width=6, id='mlp-params-row')
                                            ]),
                                            dbc.Button(
                                                "Train Model",
                                                id="train-btn",
                                                color="primary",
                                                className="w-100 mt-3"
                                            ),
                                            html.Div(id='model-summary', style={'margin-top': '15px'})
                                        ])
                                    ]),
                                    label="Configuration"
                                ),
                                dbc.Tab(
                                    dbc.Container(fluid=True, children=[
                                        dbc.Row([
                                            dbc.Col(dcc.Graph(id='confusion-matrix'), width=6),
                                            dbc.Col(dcc.Graph(id='feature-importance'), width=6)
                                        ], style={'margin-bottom': '15px'}),
                                        dbc.Row([
                                            dbc.Col(dcc.Graph(id='roc-curve'), width=6),
                                            dbc.Col(dcc.Graph(id='pr-curve'), width=6)
                                        ])
                                    ]),
                                    label="Results"
                                )
                            ])
                        ], style=section_style)
                    ], width=12)
                ], style={'margin-top': '15px'})
            ], width=12)
        ])
    ], style=dashboard_style)