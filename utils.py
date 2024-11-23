import plotly.graph_objects as go

def create_gauge_chart(probability, width=400, height=300, bg_color="rgba(0, 0, 0, 0)", font_color="white"):
    """
    Create a gauge chart to represent churn probability.

    Parameters:
        probability (float): Probability value between 0 and 1.
        width (int): Width of the chart.
        height (int): Height of the chart.
        bg_color (str): Background color of the chart.
        font_color (str): Font color of the text.
    """
    # Determine the color based on the probability range
    if probability < 0.3:
        color = "rgba(0, 255, 0, 0.8)"  # Green for low churn risk
    elif probability < 0.6:
        color = "rgba(255, 255, 0, 0.8)"  # Yellow for medium churn risk
    else:
        color = "rgba(255, 0, 0, 0.8)"  # Red for high churn risk

    # Create the gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Probability", 'font': {'color': font_color}},
            number={'font': {'color': font_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': font_color},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                    {'range': [30, 60], 'color': "rgba(255, 255, 0, 0.3)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': font_color, 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        )
    )

    # Update layout
    fig.update_layout(
        paper_bgcolor=bg_color,
        font={'color': font_color},
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_model_probability_chart(probabilities, width=600, height=400, font_color="white", bar_color="rgba(0, 123, 255, 0.8)"):
    """
    Create a horizontal bar chart to compare churn probabilities by model.

    Parameters:
        probabilities (dict): Dictionary of model names and probabilities.
        width (int): Width of the chart.
        height (int): Height of the chart.
        font_color (str): Font color of the text.
        bar_color (str): Color of the bars.
    """
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    # Create the bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation='h',
                text=[f'{p:.2%}' for p in probs],
                textposition='auto',
                marker=dict(color=bar_color)
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title='Churn Probability by Model',
        yaxis_title='Models',
        xaxis_title='Probability',
        xaxis=dict(tickformat='.0%', range=[0, max(1, max(probs) + 0.1)]),  # Dynamic scaling
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=font_color)
    )
    return fig


def create_percentile_bar_chart(percentiles, width=600, height=400, font_color="white", bar_color="rgba(0, 123, 255, 0.8)"):
    """
    Create a vertical bar chart to represent customer percentiles across metrics.

    Parameters:
        percentiles (dict): Dictionary of metrics and their percentile values.
        width (int): Width of the chart.
        height (int): Height of the chart.
        font_color (str): Font color of the text.
        bar_color (str): Color of the bars.
    """
    metrics = list(percentiles.keys())
    values = list(percentiles.values())

    # Create the bar chart
    fig = go.Figure(
        go.Bar(
            x=metrics,
            y=values,
            text=[f"{v}%" for v in values],
            textposition='auto',
            marker=dict(color=bar_color)
        )
    )

    # Update layout
    fig.update_layout(
        title="Customer Percentiles Across Metrics",
        xaxis_title="Metrics",
        yaxis_title="Percentile (%)",
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=font_color)
    )
    return fig
