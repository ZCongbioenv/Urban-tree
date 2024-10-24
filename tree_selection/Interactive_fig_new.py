import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from flask_caching import Cache
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def get_color_scale_and_range(index, data):
    if index == 'TVDI':
        classes = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#f46d43', '#d73027', '#a50026']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(classes, len(colors))
        return cmap, norm
    else:
        color_scale = 'Viridis'
        color_range = [np.nanmin(data), np.nanmax(data)]
        if index in ['LST', 'SWIR1', 'TIRS1', 'SWIR2']:
            color_scale = 'Inferno'
        elif index in ['NDVI', 'EVI2', 'MSAVI2', 'GNDVI', 'NDMI']:
            color_scale = 'YlGn'
        elif index in ['NDWI', 'MNDWI', 'NDSI2', 'MSI']:
            color_scale = 'RdYlGn_r'
        return color_scale, color_range

def grid_to_coords(grid_height=1327, grid_width=810):
    ll = [11.80442, 57.46971]
    ul = [11.80442, 57.81813]
    ur = [12.18235, 57.81813]
    lr = [12.18235, 57.46971]

    x = np.linspace(0, 1, grid_width)
    y = np.linspace(0, 1, grid_height)
    xv, yv = np.meshgrid(x, y)

    latitudes = ul[1] + (ll[1] - ul[1]) * yv + (ur[1] - ul[1]) * xv
    longitudes = ll[0] + (lr[0] - ll[0]) * xv + (ul[0] - ll[0]) * yv

    latitudes = np.flip(latitudes, axis=0)
    return latitudes, longitudes

latitudes, longitudes = grid_to_coords()

def create_map(data, index, year, season):
    logging.debug(f"Creating map for index: {index}, year: {year}, season: {season}")
    if data is not None:
        if index == 'TVDI':
            classes = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#f46d43', '#d73027', '#a50026']
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(classes, len(colors))

            # Prepare color scale for Plotly
            color_scale = [
                [0.0, colors[0]], [0.4, colors[0]],
                [0.4, colors[1]], [0.5, colors[1]],
                [0.5, colors[2]], [0.6, colors[2]],
                [0.6, colors[3]], [0.7, colors[3]],
                [0.7, colors[4]], [0.8, colors[4]],
                [0.8, colors[5]], [0.9, colors[5]],
                [0.9, colors[6]], [1.0, colors[6]],
            ]

            fig = go.Figure(data=go.Heatmap(
                z=np.flipud(data),
                x=longitudes[0],
                y=latitudes[:, 0],
                colorscale=color_scale,
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title='TVDI',
                    tickvals=classes,
                    ticktext=[str(c) for c in classes]
                )
            ))

            # Add pins for specific locations
            pins = {
                "Natrium": (57.6856, 11.9598),
                "Gothenburg C": (57.7086, 11.9733),
                "Lindholmen": (57.7078, 11.9383),
                "Botanical Garden": (57.6829, 11.9504),
                "stora delsjön":(57.6846,12.0479),
                "Västra Frölunda":(57.6529,11.9106),
                "Sandsjöbacka Naturreservat":(57.57007372846006, 12.009118184656508),
                "Liseberg":(57.6952,11.9925),
                "Bridge Weather Station": (57.691201,11.901476),
                "SMHI Station": (57.7156,11.9924)
            }

            for place, (lat, lon) in pins.items():
                text_position = 'middle right'
                symbol = 'triangle-up'
                if place == "Botanical Garden":
                    text_position = 'bottom center'
                elif place == "Lindholmen":
                    text_position = 'top center'
                elif place == "Bridge Weather Station":
                    symbol = 'circle'
                elif place == "SMHI Station":
                    symbol = 'circle'
                
                fig.add_trace(go.Scatter(
                    x=[lon],
                    y=[lat],
                    text=[place],
                    mode='markers+text',
                    marker=dict(
                        size=16,
                        color='black',
                        symbol=symbol  # Marker shape set to triangle
                    ),
                    textposition=text_position,
                    textfont=dict(
                        color='blue'  # Text color set to blue
                    ),
                    showlegend=False  # Hide legend for pins
                ))

            fig.update_layout(
                title=f"TVDI - {year} {season}",
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                width=800,
                height=1000
            )

            return dcc.Graph(figure=fig)
        else:
            color_scale, color_range = get_color_scale_and_range(index, data)
            fig = go.Figure(data=go.Heatmap(
                z=np.flipud(data),
                x=longitudes[0],
                y=latitudes[:, 0],
                colorscale=color_scale,
                zmin=color_range[0],
                zmax=color_range[1],
                colorbar=dict(
                    title=index
                )
            ))

            # Add pins for specific locations
            pins = {
                "Natrium": (57.6856, 11.9598),
                "Gothenburg C": (57.7086, 11.9733),
                "Lindholmen": (57.7078, 11.9383),
                "Botanical Garden": (57.6829, 11.9504),                
                "stora delsjön":(57.6846,12.0479),
                "Västra Frölunda":(57.6529,11.9106),
                "Sandsjöbacka Naturreservat":(57.57007372846006, 12.009118184656508),
                "Liseberg":(57.6952,11.9925),
                "Bridge Weather Station": (57.691201,11.901476),
                "SMHI Station": (57.7156,11.9924)
            }

            for place, (lat, lon) in pins.items():
                text_position = 'middle right'
                symbol = 'triangle-up'
                if place == "Botanical Garden":
                    text_position = 'bottom center'
                elif place == "Lindholmen":
                    text_position = 'top center'
                elif place == "Bridge Weather Station":
                    symbol = 'circle'
                elif place == "SMHI Station":
                    symbol = 'circle'
                fig.add_trace(go.Scatter(
                    x=[lon],
                    y=[lat],
                    text=[place],
                    mode='markers+text',
                    marker=dict(
                        size=16,
                        color='black',
                        symbol=symbol  # Marker shape set to triangle
                    ),
                    textposition=text_position,
                    textfont=dict(
                        color='blue'  # Text color set to blue
                    ),
                    showlegend=False  # Hide legend for pins
                ))

            fig.update_xaxes(title_text='Longitude', range=[longitudes.min(), longitudes.max()])
            fig.update_yaxes(title_text='Latitude', range=[latitudes.min(), latitudes.max()])
            fig.update_layout(width=800, height=1000)
            return dcc.Graph(figure=fig)
    else:
        logging.debug(f"No data available for index: {index}, year: {year}, season: {season}")
        return html.Div("Data not available")


indices = ['TIRS1', 'SWIR2', 'SWIR1', 'NIR', 'SAVI', 'NDVI', 'MSAVI2', 'NDSI2', 'MSI', 'NDWI', 'GNDVI', 'EVI2', 'LST', 'NDMI', 'MNDWI', 'TVDI']
seasons = ['spring', 'summer', 'fall']
years = range(2014, 2024)

app = dash.Dash(__name__)
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

app.layout = html.Div([
    html.H1("Interactive Map of Gothenburg for Indices Visualization (LandSat 8 SR, 30m, 2014-2023)"),
    html.Div([
        html.Div([
            html.H3("Map 1"),
            dcc.Dropdown(
                id='index1-dropdown',
                options=[{'label': idx, 'value': idx} for idx in indices],
                value='NDVI',
                style={'width': '90%', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='year1-dropdown',
                options=[{'label': str(year), 'value': year} for year in years],
                value=2014,
                style={'width': '90%', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='season1-dropdown',
                options=[{'label': season, 'value': season} for season in seasons],
                value='summer',
                style={'width': '90%', 'display': 'inline-block'}
            ),
            html.Div(id='index-map1')
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
        html.Div([
            html.H3("Map 2"),
            dcc.Dropdown(
                id='index2-dropdown',
                options=[{'label': idx, 'value': idx} for idx in indices],
                value='NDVI',
                style={'width': '90%', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='year2-dropdown',
                options=[{'label': str(year), 'value': year} for year in years],
                value=2015,
                style={'width': '90%', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='season2-dropdown',
                options=[{'label': season, 'value': season} for season in seasons],
                value='summer',
                style={'width': '90%', 'display': 'inline-block'}
            ),
            html.Div(id='index-map2')
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
    ]),
    html.Hr(),
    html.Div([
        html.H2("Trend Analysis"),
        dcc.Dropdown(
            id='trend-index-dropdown',
            options=[{'label': idx, 'value': idx} for idx in indices],
            value=['NDVI'],
            multi=True,
            style={'width': '50%', 'display': 'inline-block'}
        ),
        dcc.Graph(id='trend-graph')
    ]),
    html.Hr(),
    html.Div([
        html.H2("Correlation Analysis"),
        dcc.Dropdown(
            id='correlation-index1-dropdown',
            options=[{'label': idx, 'value': idx} for idx in indices],
            value='NDVI',
            style={'width': '30%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='correlation-index2-dropdown',
            options=[{'label': idx, 'value': idx} for idx in indices],
            value='LST',
            style={'width': '30%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='correlation-year-dropdown',
            options=[{'label': str(year), 'value': year} for year in years],
            value=2014,
            style={'width': '20%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='correlation-season-dropdown',
            options=[{'label': season, 'value': season} for season in seasons],
            value='summer',
            style={'width': '20%', 'display': 'inline-block'}
        ),
        dcc.Graph(id='correlation-graph')
    ]),
    html.Hr(),
    html.Div([
        html.H2("Data Summary"),
        html.Div(id='data-summary')
    ]),
    html.Hr(),
    html.Div([
        html.P("Done by Zhicong using Python Dash and Plotly with ChatGPT",
               style={'color': 'blue', 'font-family': 'Courier New', 'font-size': '16px'})
    ])
])

@cache.memoize(timeout=300)
def load_data(index, year, season):
    data_path = f'./Landsat8_SR_{index}/{index}_{year}_{season}_mean.npy'
    if os.path.exists(data_path):
        logging.debug(f"Loading data from {data_path}")
        return np.load(data_path)
    logging.debug(f"Data file {data_path} does not exist")
    return None

@app.callback(
    [Output('index-map1', 'children'),
     Output('index-map2', 'children'),
     Output('trend-graph', 'figure'),
     Output('correlation-graph', 'figure'),
     Output('data-summary', 'children')],
    [Input('index1-dropdown', 'value'),
     Input('year1-dropdown', 'value'),
     Input('season1-dropdown', 'value'),
     Input('index2-dropdown', 'value'),
     Input('year2-dropdown', 'value'),
     Input('season2-dropdown', 'value'),
     Input('trend-index-dropdown', 'value'),
     Input('correlation-index1-dropdown', 'value'),
     Input('correlation-index2-dropdown', 'value'),
     Input('correlation-year-dropdown', 'value'),
     Input('correlation-season-dropdown', 'value')]
)
def update_graph(index1, year1, season1, index2, year2, season2, trend_indices, corr_index1, corr_index2, corr_year, corr_season):
    logging.debug("Entered update_graph callback")
    try:
        logging.debug(f"Inputs - index1: {index1}, year1: {year1}, season1: {season1}, index2: {index2}, year2: {year2}, season2: {season2}, trend_indices: {trend_indices}, corr_index1: {corr_index1}, corr_index2: {corr_index2}, corr_year: {corr_year}, corr_season: {corr_season}")
        
        data1 = load_data(index1, year1, season1)
        data2 = load_data(index2, year2, season2)
        corr_data1 = load_data(corr_index1, corr_year, corr_season)
        corr_data2 = load_data(corr_index2, corr_year, corr_season)

        logging.debug("Data loaded successfully")

        fig1 = create_map(data1, index1, year1, season1)
        fig2 = create_map(data2, index2, year2, season2)

        trend_fig = go.Figure()
        for idx in trend_indices:
            trend_data = []
            for year in years:
                yearly_data = load_data(idx, year, season1)
                if yearly_data is not None:
                    trend_data.append(np.nanmean(yearly_data))

            if trend_data:
                trend_fig.add_trace(go.Scatter(
                    x=list(years),
                    y=trend_data,
                    mode='lines+markers',
                    name=idx
                ))
        trend_fig.update_layout(
            title=f'Trend Analysis for Selected Indices ({season1})',
            xaxis_title='Year',
            yaxis_title='Mean Value'
        )

        correlation_fig = go.Figure()
        if corr_data1 is not None and corr_data2 is not None:
            mask = ~np.isnan(corr_data1) & ~np.isnan(corr_data2)
            corr_data1 = corr_data1[mask]
            corr_data2 = corr_data2[mask]

            if len(corr_data1) > 0 and len(corr_data2) > 0:
                model = LinearRegression().fit(corr_data1.reshape(-1, 1), corr_data2)
                r2 = model.score(corr_data1.reshape(-1, 1), corr_data2)
                r2_text = f'R² = {r2:.2f}'
            else:
                r2_text = 'R² = N/A'

            correlation_fig = px.scatter(
                x=corr_data1,
                y=corr_data2,
                labels={'x': corr_index1, 'y': corr_index2},
                title=f'Correlation between {corr_index1} and {corr_index2} ({corr_year} {corr_season})'
            )
            correlation_fig.add_annotation(
                x=max(corr_data1),
                y=min(corr_data2),
                text=r2_text,
                showarrow=False,
                font=dict(size=12, color='red')
            )
            correlation_fig.update_layout(width=800, height=600)

        summary = "Data not available"
        if data1 is not None:
            mean_val = np.nanmean(data1)
            median_val = np.nanmedian(data1)
            std_val = np.nanstd(data1)
            min_val = np.nanmin(data1)
            max_val = np.nanmax(data1)
            summary = html.Div([
                html.H4(f"Summary for {index1} ({year1} {season1})"),
                html.P(f"Mean: {mean_val:.2f}"),
                html.P(f"Median: {median_val:.2f}"),
                html.P(f"Standard Deviation: {std_val:.2f}"),
                html.P(f"Min: {min_val:.2f}"),
                html.P(f"Max: {max_val:.2f}")
            ])

        logging.debug("Returning figures and summary")
        return fig1, fig2, trend_fig, correlation_fig, summary

    except Exception as e:
        logging.error(f"Error in update_graph callback: {e}")
        return html.Div("Error loading data"), html.Div("Error loading data"), go.Figure(), go.Figure(), "Error loading data"

if __name__ == '__main__':
    app.run_server(debug=True)
