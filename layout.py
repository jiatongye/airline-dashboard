import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def create_layout(df_reduced, service_cols, fig_corr):
    return html.Div([
        # TITLE ROW
        html.Div([
            html.H1("✈️ Airline Satisfaction Dashboard", 
                    style={'text-align': 'center', 'color': '#3498db', 'fontSize': '2rem'}),
            html.P("Interactive dashboard for analyzing passenger metrics",
                   style={'text-align': 'center', 'color': '#7f8c8d'})
        ], style={'marginBottom': '20px'}),

        # MAIN CONTENT
        html.Div([
            dbc.Row([
                # DATA OVERVIEW
                dbc.Col([
                    html.Div([
                        html.H4("Data Overview", className="mb-2 text-primary"),
                        dash_table.DataTable(
                            id='raw-data-table',
                            data=df_reduced.head(10).to_dict('records'),
                            columns=[{'name': col.replace('_', ' ').title(), 'id': col} for col in df_reduced.columns],
                            style_table={'maxHeight': '300px', 'overflowY': 'auto', 'width': '100%'},
                            style_cell={'textAlign': 'left', 'padding': '6px', 'whiteSpace': 'normal'},
                            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                            page_size=10
                        )
                    ], style=section_box())
                ], width=5),

                # SERVICE ANALYSIS
                dbc.Col([
                    html.Div([
                        html.H4("Service Rating Analysis", className="mb-2 text-primary"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id='service-dropdown',
                                    options=[{'label': col.replace('_', ' ').title(), 'value': col} for col in service_cols],
                                    value='in_flight_entertainment',
                                    style={'marginBottom': '10px'}
                                ),
                                dcc.Dropdown(
                                    id='chart-type',
                                    options=[
                                        {'label': 'Boxplot', 'value': 'box'},
                                        {'label': 'Grouped Bar Chart', 'value': 'bar'}
                                    ],
                                    value='box',
                                    style={'marginBottom': '10px'}
                                ),
                                dash_table.DataTable(
                                    id='filtered-table',
                                    columns=[], data=[],
                                    style_table={'maxHeight': '150px', 'overflowY': 'auto', 'width': '100%'},
                                    style_cell={'textAlign': 'left', 'padding': '6px', 'minWidth': '100px'},
                                    page_size=5
                                )
                            ], width=6),

                            dbc.Col([
                                dcc.Graph(id='service-boxplot', style={'height': '300px', 'width': '100%'})
                            ], width=6)
                        ])
                    ], style=section_box())
                ], width=7)
            ], className='mb-4'),

            # PREDICTIVE MODEL SECTION
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Predictive Model", className="mb-2 text-primary"),
                        dbc.Tabs([
                            dbc.Tab(
                                dbc.Card([
                                    dbc.CardBody([
                                        model_config_row(),
                                        model_param_row(),
                                        dbc.Button("Train Model", id="train-btn", color="primary", className="w-100 mt-3"),
                                        html.Div(id='model-summary', style={'marginTop': '15px'})
                                    ])
                                ]),
                                label="Configuration"
                            ),
                            dbc.Tab(
                                dbc.Container(fluid=True, children=[
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id='confusion-matrix'), width=6),
                                        dbc.Col(dcc.Graph(id='feature-importance'), width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id='roc-curve'), width=6),
                                        dbc.Col(dcc.Graph(id='pr-curve'), width=6)
                                    ])
                                ]),
                                label="Results"
                            )
                        ])
                    ], style=section_box())
                ])
            ])
        ], style={'maxWidth': '1300px', 'margin': '0 auto'})
    ])

def section_box():
    return {
        'border': '1px solid #dee2e6',
        'borderRadius': '10px',
        'padding': '15px',
        'marginBottom': '15px',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'backgroundColor': 'white'
    }

def model_config_row():
    return dbc.Row([
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
            dbc.Input(id='n-estimators', type='number', value=100, min=10, max=1000, step=10)
        ], width=3),
        dbc.Col([
            html.Label("Learning Rate"),
            dbc.Input(id='learning-rate', type='number', value=0.1, min=0.001, max=1, step=0.01)
        ], width=3),
        dbc.Col([
            html.Label("Max Depth"),
            dbc.Input(id='max-depth', type='number', value=6, min=1, max=20, step=1)
        ], width=3)
    ])

def model_param_row():
    return dbc.Row([
        dbc.Col([
            html.Label("Min Samples Split"),
            dbc.Input(id='min-samples-split', type='number', value=2, min=2, max=20, step=1)
        ], width=3),
        dbc.Col([
            html.Label("Min Samples Leaf"),
            dbc.Input(id='min-samples-leaf', type='number', value=1, min=1, max=20, step=1)
        ], width=3),
        dbc.Col([
            html.Label("Hidden Layer Sizes"),
            dbc.Input(id='hidden-layer-sizes', type='text', value="(100,)", placeholder="e.g., (100,50)")
        ], width=6, id='mlp-params-row')
    ])
