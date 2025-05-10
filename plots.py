import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def create_confusion_matrix(cm, accuracy):
    """Create annotated confusion matrix heatmap"""
    labels = ["Neutral/Dissatisfied", "Satisfied"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            hoverinfo="z"
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix (Accuracy: {accuracy:.2%})",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(family="Arial", size=12)
    )
    return fig

def create_feature_importance(importance, features, model_type):
    """Create horizontal bar chart of feature importance"""
    df_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(10)
    
    title = "Top Features"
    if model_type == 'logreg':
        title += " (Logistic Regression Coefficients)"
        color_col = 'Importance'
    else:
        title += " (Gradient Boosting Importance)"
        color_col = 'Importance'
    
    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color=color_col,
        color_continuous_scale='Tealrose'
    )
    fig.update_layout(
        height=400,
        margin=dict(l=100, r=50, t=80, b=50),
        font=dict(family="Arial", size=12)
    )
    return fig

def create_roc_curve(fpr, tpr, roc_auc):
    """Create ROC curve visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#219EBC', width=3),
        name=f'ROC (AUC = {roc_auc:.2f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#FFB703', dash='dash', width=2),
        name='Random Chance'
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_pr_curve(precision, recall, pr_auc):
    """Create precision-recall curve visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        line=dict(color='#FB8500', width=3),
        name=f'PR (AUC = {pr_auc:.2f})'
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_boxplot(df, service_col):
    """Create boxplot for service ratings"""
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
    """Create grouped barchart for service ratings"""
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