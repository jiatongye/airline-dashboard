import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import pandas as pd
from data_cleaning import clean_data
from model import train_model
from layout import create_layout
from callbacks import register_callbacks

# Initialize app with callback exceptions suppressed
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.LUX],
               suppress_callback_exceptions=True)
app.title = "Airline Satisfaction Analytics"

# Load and clean data
df, df_reduced = clean_data()
service_cols = ['ease_of_online_booking', 'check_in_service', 'online_boarding',
               'on_board_service', 'in_flight_service', 'in_flight_entertainment']

# Generate correlation figure
corr_matrix = pd.get_dummies(df_reduced).corr().round(2)
fig_corr = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    colorscale='YlGnBu',
    annotation_text=corr_matrix.values.round(2),
    showscale=True,
    font_colors=['black', 'white']
)
fig_corr.update_layout(
    title="Correlation Heatmap (All Features)",
    font=dict(family='Arial', size=10),
    margin=dict(t=100, l=150, r=50, b=150),
    height=700,
    xaxis=dict(tickangle=45)
)

# Set up layout
app.layout = create_layout(df_reduced, service_cols, fig_corr)

# Register callbacks
register_callbacks(app, df)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
