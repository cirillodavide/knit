from sklearn.manifold import TSNE
from bokeh.plotting import *
from bokeh.models import HoverTool 
from scipy.stats import rankdata
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

filename = 'out/coexpr_user-similarity.csv'
embedded_coords = pd.read_csv(filename,sep=',',skipinitialspace=True,index_col=0)

xycoords = TSNE().fit_transform(embedded_coords)

plot = figure(plot_width=900, plot_height=700, title="Terms Map by t-SNE",
       tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
       x_axis_type=None, y_axis_type=None, min_border=1)
plot.scatter(
    x = xycoords[:,0],
    y = xycoords[:,1])

plot.select(HoverTool).tooltips = {"/r/":"@subreddit"}
show(plot) # command: jupyter notebook
