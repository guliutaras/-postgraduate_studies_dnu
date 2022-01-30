from neural_network import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openml.datasets.functions import get_dataset
from gtda.plotting import plot_point_cloud
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from utils import samplefuntion2d
from utils import func_sin
from utils import func_complex

random.seed(0)

#point_cloud = get_dataset(42182).get_data(dataset_format='array')[0]
#fig = plot_point_cloud(point_cloud)
#fig.show()

amount_totrain = 1000
amount_totest = 500
start_sample = -100
end_sample = 100

totalamount = amount_totrain+amount_totest + 5 

sinewave = samplefuntion2d(func_complex, start_sample , end_sample , totalamount)

minmax_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
data_minmax = minmax_scaler.fit_transform(sinewave.reshape(-1,1))

x1 = data_minmax[:amount_totrain].tolist()
X2 = data_minmax[1:amount_totrain+1].tolist()
x3 = data_minmax[2:amount_totrain+2].tolist()
x4 = data_minmax[3:amount_totrain+3].tolist()
dfx =  list(map(lambda x, y, z, k:[(x[0],y[0],z[0]),[k[0]]], x1, X2, x3, x4))

x1test = data_minmax[amount_totrain:amount_totrain+amount_totest].tolist()
X2test = data_minmax[amount_totrain+1:amount_totrain+amount_totest+1].tolist()
x3test = data_minmax[amount_totrain+2:amount_totrain+amount_totest+2].tolist()
x4test = data_minmax[amount_totrain+3:amount_totrain+amount_totest+3].tolist()
dfxtest =  list(map(lambda x, y, z, k:[(x[0],y[0],z[0]),[k[0]]], x1test, X2test, x3test, x4test))

xground = sinewave[3:amount_totrain+amount_totest+3].tolist();

"""
df = pd.read_csv("EKG_Reading_Up16_Shift2.csv")

df1sub1 = df.iloc[:1000, 1].values.tolist()
df1sub2 = df.iloc[:1000, 2].values.tolist()
df1sub3 = df.iloc[:1000, 3].values.tolist()
df1 =  list(map(lambda x, y, z:[(x,y),[z]], df1sub2, df1sub3, df1sub1))
df2sub1 = df.iloc[1000:1500, 1].values.tolist()
df2sub2 = df.iloc[1000:1500, 2].values.tolist()
df2sub3 = df.iloc[1000:1500, 3].values.tolist()
df2 =  list(map(lambda x, y, z:[(x,y),[z]], df2sub2, df2sub3, df2sub1))
""" 

nn = NeuralNetwork(learning_rate=0.1, debug=True)
nn.add_layer(n_inputs=3, n_neurons=3)
nn.add_layer(n_inputs=3, n_neurons=1)
nn.train(dataset=dfx, n_iterations=100, print_error_report=True)

# test
actual_outputs = []
actual_outputsTrans = []
for j, (inputs, targets) in enumerate(dfx+dfxtest):
     іses = nn.feed_forward(inputs)
     actual_outputs.append(іses[0])

actual_outputsTrans= minmax_scaler.inverse_transform(np.array(actual_outputs).reshape(-1,1)).tolist()
flat_list = [item for sublist in actual_outputsTrans for item in sublist]

fig2 = px.line(title='ground')
fig2.add_scatter(x=np.linspace(start_sample , end_sample, totalamount)[3:amount_totrain+amount_totest+3],y=xground, name='XComp')
fig2.show()

fig3 = px.line(title='Actual')
fig3.add_scatter(x=np.linspace(start_sample , end_sample, totalamount)[3:amount_totrain+amount_totest+3],y=flat_list, name='XComp')
fig3.show()