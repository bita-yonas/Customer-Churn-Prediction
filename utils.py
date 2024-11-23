import plotly.graph_objects as go
import plotly.express as px

# Utility Function: Dynamic Color Generator
def generate_color_palette(n, base_color="blue"):
    """
    Generate a color palette based on Plotly's built-in color sequences.

    Parameters:
        n (int): Number of colors to generate.
        base_color (str): Base color scheme ('blue', 'green', 'red', etc.).

    Returns:
        list: List of colors.
    """
    color_sequences = px.colors.qualitative
    if base_color in color_sequences:
        return color_sequences[base_color][:n]
    return px.colors.qualitative.Plotly[:n]

# 1. Gauge Chart
def create_gauge_chart(probability, width=400, height=300, bg_color="rgba(0, 0, 0, 0)", font_color="white"):
    if probability < 0.3:
        color = "rgba(0, 255, 0, 0.8)"
    elif probability < 0.6:
        color = "rgba(255, 255, 0, 0.8)"
    else:
        color = "rgba(255, 0, 0, 0.8)"

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

    fig.update_layout(
        paper_bgcolor=bg_color,
        font={'color': font_color},
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# 2. Model Probability Chart
def create_model_probability_chart(probabilities, width=600, height=400, font_color="white", bar_color="rgba(0, 123, 255, 0.8)"):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

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

    fig.update_layout(
        title='Churn Probability by Model',
        yaxis_title='Models',
        xaxis_title='Probability',
        xaxis=dict(tickformat='.0%', range=[0, max(1, max(probs) + 0.1)]),
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=font_color)
    )
    return fig

# 3. Percentile Bar Chart
def create_percentile_bar_chart(percentiles, width=600, height=400, font_color="white", bar_color="rgba(0, 123, 255, 0.8)"):
    metrics = list(percentiles.keys())
    values = list(percentiles.values())

    fig = go.Figure(
        go.Bar(
            x=metrics,
            y=values,
            text=[f"{v}%" for v in values],
            textposition='auto',
            marker=dict(color=bar_color)
        )
    )

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

# 4. Donut Chart
def create_donut_chart(categories, values, width=500, height=400, font_color="white"):
    """
    Create a donut chart to visualize categorical data distribution.

    Parameters:
        categories (list): List of category names.
        values (list): List of corresponding values.
        width (int): Width of the chart.
        height (int): Height of the chart.
        font_color (str): Font color of the text.
    """
    fig = go.Figure(
        go.Pie(
            labels=categories,
            values=values,
            hole=0.4,  # Creates the donut effect
            textinfo='percent+label',
            hoverinfo='label+percent',
            marker=dict(colors=generate_color_palette(len(categories))),
        )
    )

    fig.update_layout(
        title="Churn vs Non-Churn Distribution",
        height=height,
        width=width,
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Example Usage
if __name__ == "__main__":
    # Gauge Chart Example
    fig1 = create_gauge_chart(0.72)
    fig1.show()

    # Model Probability Chart Example
    probabilities = {"Model A": 0.8, "Model B": 0.6, "Model C": 0.3}
    fig2 = create_model_probability_chart(probabilities)
    fig2.show()

    # Percentile Bar Chart Example
    percentiles = {"Metric 1": 80, "Metric 2": 70, "Metric 3": 60}
    fig3 = create_percentile_bar_chart(percentiles)
    fig3.show()

    # Donut Chart Example
    categories = ["Churn", "Non-Churn"]
    values = [40, 60]
    fig4 = create_donut_chart(categories, values)
    fig4.show()
