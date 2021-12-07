#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: Project4
@File   : app.py
@Author : HE SICHENG
@Time   : 2021/12/3 
"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from surprise import KNNBasic, Reader, Dataset
import warnings
warnings.filterwarnings("ignore")

def get_top_10(new_rate):
    new_rate = pd.DataFrame({'movieID': movieID, 'rate': new_rate})
    new_rate['iid'] = new_rate['movieID'].map(lambda x: trainset.to_inner_iid(x))
    iid_to_rate = {iid: rate for iid, rate in zip(new_rate['iid'], new_rate['rate'])}
    est_rate = []

    for mid, iid, rate in zip(new_rate['movieID'], new_rate['iid'], new_rate['rate']):
        if rate != 0:
            est_rate.append(-1)
            continue
        nei_rate = []
        nei_weight = []
        kneighbors = algo.get_neighbors(iid, 20)
        for nei in kneighbors:
            r = iid_to_rate[nei]
            if r == 0:
                continue
            nei_rate.append(r)
            nei_weight.append(algo.sim[iid, nei])
        if len(nei_rate) == 0:
            est_rate.append(movie_avg[mid])
        else:
            est_rate.append(np.average(nei_rate, weights=nei_weight))
    new_rate['est_rate'] = est_rate
    top10 = new_rate.sort_values(['est_rate'], ascending=False)[:10]['movieID'].tolist()
    return top10

def get_movie_src(id):
    src = "https://liangfgithub.github.io/MovieImages/{}.jpg?raw=true"
    return src.format(id)

def get_image_oneline(id_list, rank=None):
    children = []
    if rank is None:
        for id in id_list:
            ch = html.Div(style={'width': '18%', 'display': 'inline-block', "margin-right": "15px"}, children=[
                     html.Img(src=get_movie_src(id), style={"margin-bottom": "15px"}),
                     html.Div(movieid_to_name[id])
            ])
            children.append(ch)
    else:
        for id, rk in zip(id_list, rank):
            ch = html.Div(style={'width': '18%', 'display': 'inline-block', "margin-right": "15px"}, children=[
                     html.H3('Top {}'.format(rk)),
                     html.Img(src=get_movie_src(id), style={"margin-bottom": "15px"}),
                     html.Div(movieid_to_name[id])
            ])
            children.append(ch)
    return html.Div(style={'textAlign': 'center',  'align-items': 'center'}, children=children)

def get_image(id_list, rank=None):
    children = []
    if rank is None:
        for i in range(0, len(id_list) - 2, 5):
            five_list = id_list[i:i+5]
            children.append(get_image_oneline(five_list))
            children.append(html.Hr(style={"margin-bottom": "25px", "margin-top": "25px"}))
        return html.Div(style={'textAlign': 'center',  'align-items': 'center'}, children=children)
    else:
        for i in range(0, len(id_list) - 2, 5):
            five_list = id_list[i:i+5]
            children.append(get_image_oneline(five_list, rank[i:i+5]))
            children.append(html.Hr(style={"margin-bottom": "25px", "margin-top": "25px"}))
        return html.Div(style={'textAlign': 'center',  'align-items': 'center'}, children=children)




def get_image_oneline_radio(id_list):
    children = []
    for id in id_list:
        ch = html.Div(style={'width': '18%', 'display': 'inline-block', "margin-right": "15px"}, children=[
            html.Img(src=get_movie_src(id), style={"margin-bottom": "15px"}),
            html.Div(movieid_to_name[id], style={"margin-bottom": "15px"}),
            dcc.RadioItems(
                id='rating_input_{}'.format(id),
                options=[{'label': str(i), 'value': str(i)} for i in range(1, 6)],
                value=None,
                labelStyle={'display': 'inline-block'}
            )

        ])
        children.append(ch)

    return html.Div(style={'textAlign': 'center',  'align-items': 'center'}, children=children)

def get_image_radio(id_list):
    children = []
    for i in range(0, len(id_list) - 2, 5):
        five_list = id_list[i:i+5]
        children.append(get_image_oneline_radio(five_list))
        children.append(html.Hr(style={"margin-bottom": "30px", "margin-top": "30px"}))
    return html.Div(style={'textAlign': 'center',  'align-items': 'center', "margin-top": "15px",
                           "maxHeight": "650px", "overflow": "scroll"}, children=children)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

button_style = {'background-color': '#228B22',
                'font-size': '20px',
                'color': 'white',
                'height': '60px',
                'width': '250px'}

myurl = "https://liangfgithub.github.io/MovieData/"
ratings = pd.read_csv(myurl + 'ratings.dat?raw=true', header=None, sep=':').dropna(axis=1)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings.drop(['Timestamp'], axis=1, inplace=True)
movies = pd.read_csv(myurl + 'movies.dat?raw=true', header=None, sep='::', encoding='latin1')
movies.columns = ['MovieID', 'Title', 'Genres']
movieid_to_name = {}
for key, value in zip(movies['MovieID'], movies['Title']):
    movieid_to_name[key] = value
movies['Genres'] = movies['Genres'].map(lambda s: s.split('|'))
genre_list = ["Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime",
               "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical",
               "Mystery", "Romance", "Sci-Fi",
               "Thriller", "War", "Western"]
movieID = np.sort(ratings['MovieID'].unique()).tolist()
id_displaying = movieID[:50]
states = [State('rating_input_{}'.format(id), 'value') for id in id_displaying]


for g in  genre_list:
    movies[g] = movies['Genres'].map(lambda x: 1 if g in x else 0)
movie_pop = ratings.groupby(['MovieID'])['Rating'].agg(['mean', 'count']).reset_index()

genre_popular_10 = pd.DataFrame()
# geren_highrate_10 = pd.DataFrame()

for genre in genre_list:
    movie_spe_genre = movies.loc[movies[genre] == 1, ['MovieID', 'Title']]
    movie_spe_genre = pd.merge(movie_spe_genre, movie_pop, on='MovieID')
    # movie_spe_genre.sort_values(['count'], ascending=False)[:10]['MovieID'].values.tolist()

    most_popular_10 = movie_spe_genre.sort_values(['count'], ascending=False)[:10]['MovieID'].values.tolist()
    # high_rated_10 = movie_spe_genre[movie_spe_genre['count'] >= 10]
    # high_rated_10 = high_rated_10.sort_values(['mean'], ascending=False)[:10]['MovieID'].values.tolist()
    genre_popular_10[genre] = most_popular_10
    # geren_highrate_10[genre] = high_rated_10

# model
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)
sim_options = {'name': 'cosine',
               'user_based': False}
trainset = data.build_full_trainset()
algo = KNNBasic(k=20, min_k=1, sim_options=sim_options)
algo.fit(trainset)
movie_avg = {i: avg for i, avg in zip(movie_pop['MovieID'], movie_pop['mean'])}


def get_movie_src(idx):
    src = "https://liangfgithub.github.io/MovieImages/{}.jpg?raw=true"
    return src.format(idx)

def get_sys1_output():
    children = [
        html.H2('Please select your favorite genre'),
        html.Div(style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
                 children=[
                     dcc.Dropdown(
                         id='genre_dropdown',
                         options=[{'label': g, 'value': g} for g in genre_list],
                         placeholder="Select a Genre",
                         clearable=False,
                         style={'width': '350px', "margin-bottom": "15px"}
                     )
                 ]),

        html.Button('Get Recommendations', id='genre_button',
                    n_clicks=0, style = button_style),
        html.Hr(style={"margin-bottom": "25px", "margin-top": "25px"}),

        html.Div(style={'width': '100%', 'display': 'inline'}, children=[
            html.H2('Movies you might like'),
            html.Div(id='genre_output', children=None)
        ])
    ]
    return html.Div(children=children)

def get_sys2_output():
    children = [
        html.H2('Please rate some movies then click the button', style={"margin-bottom": "15px"}),
        html.H2('Wait for about 10 seconds then scroll to the bottom to see the recommendations', style={"margin-bottom": "15px"}),
        html.Button('Get Recommendations', id='rating_button',
                    n_clicks=0, style=button_style),
        get_image_radio(id_displaying),
        html.Hr(style={"margin-bottom": "25px", "margin-top": "25px"}),
        html.Div(style={'width': '100%', 'display': 'inline'}, children=[
            html.H2('Movies you might like'),
            html.Div(id='rating_output', children=None)
        ])
    ]
    return html.Div(children=children)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(style={'textAlign': 'center',  'align-items': 'center'}, children=[
    dcc.Tabs(id='system_tab', value='tab_1', children=[
            dcc.Tab(label='System1: Recommender by Genre', value='tab_1', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='System2: Recommender by Rating', value='tab_2', style=tab_style, selected_style=tab_selected_style),
        ], style=tabs_styles),

    html.Div(id='sys_output', children=None)
])


# tab callback
@app.callback(
    Output('sys_output', 'children'),
    Input('system_tab', 'value'))
def update_sys_out(system):
    if system == 'tab_1':
        return get_sys1_output()
    else:
        return get_sys2_output()

@app.callback(Output('genre_output', 'children'),
              Input('genre_button', 'n_clicks'),
              State('genre_dropdown', 'value'), prevent_initial_call=True)
def update_output(n_clicks, genre):
    return [get_image(genre_popular_10[genre].values.tolist())]

@app.callback(Output("rating_output", "children"),
              Input('rating_button', 'n_clicks'),
              states, prevent_initial_call=True)
def update_rating_output(n_clicks, *rates):
    new_rate = []
    for rate in rates:
        if rate is None:
            new_rate.append(0)
        else:
            new_rate.append(int(rate))
    new_rate = new_rate + [0] * (len(movieID) - len(new_rate))
    recommend_id = get_top_10(new_rate)
    return [get_image(recommend_id, rank=[i+1 for i in range(10)])]

if __name__ == '__main__':
    #  style={"maxHeight": "250px", "overflow": "scroll"}
    app.run_server(debug=False)
