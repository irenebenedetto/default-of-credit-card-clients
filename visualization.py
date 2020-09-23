from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di 
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import os
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import plotly.figure_factory as ff
from skimage import io
#from google.colab import files
import json
import random

def plot_bar(summary, attribute, title):

  fig = px.bar(
              summary,  
              x=summary.index,  
              y=attribute, 
              text=summary.values,
              color_continuous_scale=px.colors.qualitative.Set2,
              log_y=True,
              color_discrete_sequence = px.colors.qualitative.Set2,
              title= title,
              color=summary.index,

             )
  
  
  fig.update_traces(
      texttemplate='%{text:.2s}', 
      textposition='outside',
      
      )
  fig.update_traces(hovertemplate= attribute + ' category: %{x} <br>Occurrences: %{y}') 
  fig.update_layout(
      xaxis_title= attribute +" categories",
      yaxis_title="Number of rows",
      coloraxis_showscale=False,
      xaxis_type='category',
      width=1000,
      height=500,
      hovermode='closest',
      hoverlabel=dict(
        bgcolor="white",
      )
      
  )
  fig.show()


def plot_score_outliers(score_sample, offset, title,x_position_outlier_ann, y_position_outlier_ann ):
    fig = go.Figure()
    fig.update_layout(
        title = title,
        xaxis_title= "Training dataset point",
        yaxis_title= "Score",
        showlegend=False,
        shapes=[
            # 1st highlight during Feb 4 - Feb 6
            dict(
                type="rect",
                # x-reference is assigned to the x-values

                # y-reference is assigned to the plot paper [0,1]

                x0=0,
                y0=min(score_sample),
                x1=len(score_sample),
                y1=offset,
                fillcolor="Crimson",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=[i for i in range(0, len(score_sample))], 
            y=np.sort(score_sample),
            mode="markers",
            #text=[round(v*100, 1) for v in np.cumsum(pca.explained_variance_ratio_)],

            textposition="bottom center",
            hovertemplate = "Point index: %{x}<br>Score: %{y:.2f}",
            textfont=dict(
            size=12,
            ),
            marker=dict(
              color='rgb(229,196,148)',
              size=3,
              ),
            line = dict(
                width=4
                ),
            ),

        )

    fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=0,
                y0=offset,
                x1=len(score_sample),
                y1=offset,
                line=dict(
                    color="Crimson",
                    width=2,

                ),



    ))
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.add_annotation(
                x=x_position_outlier_ann,
                y=y_position_outlier_ann,
                text="Candidate outliers",

                #showarrow=False,
                font=dict(
                    size=18,

        )
    )

    fig.show()


def plot_scattermatrix(dimensions, title, df):
  fig = px.scatter_matrix(df,
  dimensions=dimensions,
  color='default.payment.next.month',
  color_discrete_sequence = px.colors.qualitative.Set2,
  color_continuous_scale = px.colors.qualitative.Set2,
  symbol ='default.payment.next.month',
  
  )

  fig.update_layout(
        title = title,
        coloraxis_showscale=False,
        width=1700,
        height=1700,
        legend=dict(
            yanchor="top",
            font = dict(
                size=16
                
            ),
            xanchor="right",
            
        )
              
  )
  fig.update_traces(diagonal_visible=False, 
                    showupperhalf=False,
                    marker=dict(
                              colorscale=px.colors.qualitative.Set2,
                              #showscale=False, # colors encode categorical variables
                              line_color='white', 
                              line_width=0,
                              size=2
                              ),)
  fig.show()

  import plotly.figure_factory as ff

def plot_confusion_matrix(y_true, y_pred):
  
    tp = ((y_pred == 1)&(y_true == 1)).sum()
    fp = ((y_pred == 1)&(y_true == 0)).sum()

    tn = ((y_pred == 0)&(y_true == 0)).sum()
    fn = ((y_pred == 0)&(y_true == 1)).sum()

    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)

    fpr = 1 - tnr
    fnr = 1- tpr


    #                | Positive Prediction | Negative Prediction
    # Positive Class | True Positive (TP)  | False Negative (FN)
    # Negative Class | False Positive (FP) | True Negative (TN)

    z = [[tpr, fpr],
       [fnr, tnr]]
    z_text = [['True Positive rate: ' + str(round(tpr, 2)), 'False Positive rate: ' + str(round(fpr, 2))],
            ['False Negative rate: ' + str(round(fnr, 2)), 'True Negative rate: ' + str(round(tnr, 2))]]

    x = ['1', '0']
    y =  ['1', '0']

    # set up figure 
    fig = ff.create_annotated_heatmap(
      z,
      x=x, 
      y=y, 
      annotation_text=z_text, 
      colorscale= [[0,'rgb(256,256,256)'], [1, 'rgb(27,158,119)']],
      hoverinfo = "text"
    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(size=20),
                          x=0.5,
                          y=-0.1,
                          showarrow=False,
                          text="True value",
                          xref="paper",
                          yref="paper"))
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(size=20),
                          x=-0.1,
                          y=0.5,
                          showarrow=False,
                          text="Predicted value",
                          textangle=-90,
                          xref="paper",
                          yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(
      margin=dict(t=50, l=200),
      title_text='Confusion matrix',
      font=dict(size=16)
      )

    fig.show()
    return z

def print_result(s, report):
    s_table= ""
    for name, metrics in report.items():

        if not name == 'accuracy':
            s_table += "<tr><td>" + name +"</td>"
            for metric_name, metric in metrics.items():
                if not metric_name == 'support':
                    s_table += '<td>' + str(round(metric, 2)) + '</td>'
            s_table += "</tr>"

    s_table += ""
    s_table += '<tr><td>accuracy</td><td colspan=3>' + str(round(report["accuracy"], 2)) + '</td></tr>'


    di.display_html("""

    <p style='margin-bottom: 1em;font-size:15px'>
        """ + s + """
    </p>

    <table  id="customers">
        <thead>

            <tr class="second">
            <th></th>
            <th>Precision</th>
            <th>Recall</th>
                <th>F1-score</th>

            </tr>
        </thead>
        <tbody> 
            """ + s_table+ """
        </tbody>
    </table>
    """, raw=True)