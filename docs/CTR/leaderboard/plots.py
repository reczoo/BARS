import pandas as pd
from itables import show
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itables.options as opt
opt.columnDefs = [{"className": "dt-center", "targets": "_all"}]

def show_table(csv_path):
    df = pd.read_csv(csv_path)
    df["Model"] = df.apply(lambda x: f"<a href={x['Paper URL']}>{x['Model']}</a>", axis=1)
    del df["Paper URL"]
    df["Running Steps"] = df["Running Steps"].map(lambda x: f"<a href={x}>🔗</a>")
    df = df.sort_values(by=["AUC"], ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df[['AUC', 'Logloss']] = df[['AUC', 'Logloss']].applymap('{:.4f}'.format)
    show(df, lengthMenu=[10, 20, 50, 100], classes="display")

def show_plot(csv_path):
    df = pd.read_csv(csv_path).sort_values(by="AUC", ascending=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["Model"], y=df["AUC"], name="AUC", mode='lines+markers',
                   line=dict(color="#0071a7", shape="spline", smoothing=1.3), marker=dict(size=7)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["Model"], y=df["Logloss"], name="Logloss", mode='lines+markers',
                   line=dict(color="#ff404e", shape="spline", smoothing=1.3), marker=dict(size=7)),
        secondary_y=True,
    )

    fig.update_layout(
        title="Sorted benchmarking results by AUC",
        title_x=0.5,
        plot_bgcolor='white',
        autosize=True,
        width=890,
        height=450,
        legend=dict(orientation="h", x=0.4, y=-0.4)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor='lightgrey',
        secondary_y=False,
        title_text="AUC"
    )
    fig.update_yaxes(
        showgrid=False,
        gridcolor='lightgrey',
        secondary_y=True,
        title_text="Logloss"
    )
    fig.show()
