from dash import Input, Output, State, html
import dash.exceptions
import dash_bootstrap_components as dbc
import plotly.express as px
from model import train_model
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plots import (
    create_confusion_matrix,
    create_feature_importance,
    create_roc_curve,
    create_pr_curve
)

def register_callbacks(app, df):
    @app.callback(
        [
            Output('confusion-matrix', 'figure'),
            Output('feature-importance', 'figure'),
            Output('roc-curve', 'figure'),
            Output('pr-curve', 'figure'),
            Output('model-summary', 'children'),
            Output('learning-rate', 'disabled'),
            Output('max-depth', 'disabled'),
            Output('min-samples-split', 'disabled'),
            Output('min-samples-leaf', 'disabled'),
            Output('hidden-layer-sizes', 'disabled'),
            Output('mlp-params-row', 'style')
        ],
        [Input('train-btn', 'n_clicks')],
        [
            State('model-type', 'value'),
            State('n-estimators', 'value'),
            State('learning-rate', 'value'),
            State('max-depth', 'value'),
            State('min-samples-split', 'value'),
            State('min-samples-leaf', 'value'),
            State('hidden-layer-sizes', 'value')
        ]
    )
    def update_model(n_clicks, model_type, n_estimators, learning_rate, max_depth,
                   min_samples_split, min_samples_leaf, hidden_layer_sizes):
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate

        # Convert hidden_layer_sizes from string to tuple if needed
        if hidden_layer_sizes and isinstance(hidden_layer_sizes, str):
            try:
                hidden_layer_sizes = eval(hidden_layer_sizes)
            except:
                hidden_layer_sizes = (100,)

        try:
            model_results = train_model(
                model_type=model_type,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                hidden_layer_sizes=hidden_layer_sizes
            )
        except Exception as e:
            print(f"Error training model: {e}")
            return (
                go.Figure(),  # confusion-matrix
                go.Figure(),  # feature-importance
                go.Figure(),  # roc-curve
                go.Figure(),  # pr-curve
                html.Div("Error training model", className="alert alert-danger"),
                True,  # learning-rate disabled
                True,  # max-depth disabled
                True,  # min-samples-split disabled
                True,  # min-samples-leaf disabled
                True,  # hidden-layer-sizes disabled
                {'display': 'none'}  # mlp-params-row style
            )

        # Create visualizations with error handling
        try:
            confusion_fig = create_confusion_matrix(model_results['cm'], model_results['accuracy'])
        except:
            confusion_fig = go.Figure()
            confusion_fig.update_layout(title="Could not create confusion matrix")

        try:
            if model_results.get('importance') is not None:
                importance_fig = create_feature_importance(
                    model_results['importance'],
                    model_results['feature_names'],
                    model_type
                )
            else:
                importance_fig = go.Figure()
                importance_fig.update_layout(title="Feature importance not available")
        except:
            importance_fig = go.Figure()
            importance_fig.update_layout(title="Error creating feature importance")

        try:
            if all(k in model_results for k in ['fpr', 'tpr', 'roc_auc']):
                roc_fig = create_roc_curve(
                    model_results['fpr'],
                    model_results['tpr'],
                    model_results['roc_auc']
                )
                pr_fig = create_pr_curve(
                    model_results['precision'],
                    model_results['recall'],
                    model_results['pr_auc']
                )
            else:
                roc_fig = go.Figure()
                roc_fig.update_layout(title="ROC curve not available")
                pr_fig = go.Figure()
                pr_fig.update_layout(title="PR curve not available")
        except:
            roc_fig = go.Figure()
            roc_fig.update_layout(title="Error creating ROC curve")
            pr_fig = go.Figure()
            pr_fig.update_layout(title="Error creating PR curve")

        # Determine which inputs to disable
        disable_learning_rate = model_type not in ['gbc', 'xgb']
        disable_tree_params = model_type not in ['gbc', 'rf', 'xgb']
        disable_nn_params = model_type != 'mlp'

        return (
            confusion_fig,
            importance_fig,
            roc_fig,
            pr_fig,
            create_model_summary(model_results, model_type, n_estimators, learning_rate),
            disable_learning_rate,
            disable_tree_params,
            disable_tree_params,
            disable_tree_params,
            disable_nn_params,
            {'display': 'none'} if disable_nn_params else {}
        )

    @app.callback(
        [Output('filtered-table', 'data'),
         Output('filtered-table', 'columns'),
         Output('service-boxplot', 'figure')],
        [Input('service-dropdown', 'value'),
         Input('chart-type', 'value')]
    )
    def update_service_charts(service_col, chart_type):
        if service_col is None:
            raise PreventUpdate
            
        filtered_df = df[[service_col, 'satisfaction_binary']].copy()
        filtered_df['satisfaction'] = filtered_df['satisfaction_binary'].map(
            {0: 'Neutral/Dissatisfied', 1: 'Satisfied'})
        
        # Create table data
        table_data = filtered_df.to_dict('records')
        table_columns = [{'name': col.replace('_', ' ').title(), 'id': col} 
                        for col in filtered_df.columns]
        
        # Create appropriate chart
        try:
            if chart_type == 'box':
                fig = create_boxplot(filtered_df, service_col)
            else:
                fig = create_barchart(filtered_df, service_col)
        except Exception as e:
            print(f"Error creating chart: {e}")
            fig = go.Figure()
            fig.update_layout(title="Error creating visualization")
            
        return table_data, table_columns, fig

def create_model_summary(results, model_type, n_estimators, learning_rate):
    """Helper function to create model summary cards"""
    model_names = {
        'logreg': 'Logistic Regression',
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'mlp': 'Neural Network'
    }
    
    return [
        html.H4(f"{model_names.get(model_type, 'Model')} Results", 
               className="text-center mb-3"),
        dbc.Row([
            dbc.Col(create_metric_card("Accuracy", f"{results['accuracy']:.2%}", "bg-success"), width=4),
            dbc.Col(create_metric_card("ROC AUC", f"{results.get('roc_auc', 'N/A'):.3f}" if results.get('roc_auc') is not None else "N/A", "bg-info"), width=4),
            dbc.Col(create_metric_card("PR AUC", f"{results.get('pr_auc', 'N/A'):.3f}" if results.get('pr_auc') is not None else "N/A", "bg-warning"), width=4)
        ]),
        html.P(f"Trained with {n_estimators} {'iterations' if model_type == 'logreg' else 'estimators'}" if model_type != 'mlp' else "Trained with Neural Network",
              className="text-center mt-3"),
        html.P(f"{'Learning rate: ' + str(learning_rate) if model_type in ['gbc', 'xgb'] else ''}",
              className="text-center")
    ]

def create_metric_card(title, value, color_class):
    """Helper to create consistent metric cards"""
    return dbc.Card([
        dbc.CardHeader(title, className=f"{color_class} text-white"),
        dbc.CardBody([html.H2(value, className="text-center")])
    ])

def create_boxplot(df, service_col):
    """Create boxplot visualization"""
    return px.box(
        df,
        x='satisfaction',
        y=service_col,
        color='satisfaction',
        points='all',
        labels={service_col: service_col.replace('_', ' ').title()},
        color_discrete_map={'Neutral/Dissatisfied': '#FFB703', 'Satisfied': '#219EBC'}
    ).update_layout(base_layout())

def create_barchart(df, service_col):
    """Create barchart visualization"""
    return px.histogram(
        df,
        x=service_col,
        color='satisfaction',
        barmode='group',
        labels={service_col: service_col.replace('_', ' ').title()},
        color_discrete_map={'Neutral/Dissatisfied': '#FFB703', 'Satisfied': '#219EBC'}
    ).update_layout(base_layout())

def base_layout():
    """Shared layout configuration for all plots"""
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': "Arial", 'size': 12},
        'margin': dict(l=50, r=50, t=80, b=50),
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    }