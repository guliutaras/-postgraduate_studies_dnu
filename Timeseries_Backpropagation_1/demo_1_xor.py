from neural_network import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from openml.datasets.functions import get_dataset
from gtda.plotting import plot_point_cloud
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from utils import samplefuntion2d
from utils import func_sin
from utils import func_complex
from utils import func_ode_sample
from utils import plot3Dref
random.seed(0)

# point_cloud = get_dataset(42182).get_data(dataset_format='array')[0]
# fig = plot_point_cloud(point_cloud)
# fig.show()

amount_totrain = 5000
amount_totest = 2000
start_sample = -10
end_sample = 100
sample_rate = 10000
start_vec = [0,4.5,5.0002]
totalamount = amount_totrain+amount_totest + 5 

#sinewave = samplefuntion2d(func_complex, start_sample , end_sample , totalamount)
chell_sample = func_ode_sample(start_sample,end_sample,sample_rate,start_vec[0],start_vec[1],start_vec[2])

minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

minmaxed = minmax_scaler.fit_transform(chell_sample)
plot3Dref(minmaxed)

x_train = minmaxed[:amount_totrain,0].tolist()
y_train = minmaxed[:amount_totrain,1].tolist()
z_train = minmaxed[:amount_totrain,2].tolist()
x_recive = minmaxed[1:amount_totrain+1,0].tolist()
y_recive = minmaxed[1:amount_totrain+1,1].tolist()
z_recive = minmaxed[1:amount_totrain+1,2].tolist()
dfx =  list(map(lambda xt, yt, zt, xr,yr,zr:[(xt,yt,zt),[xr,yr,zr]], x_train,y_train, z_train, x_recive,y_recive,z_recive))

x_test_t = minmaxed[amount_totrain:amount_totrain+amount_totest,0].tolist()
y_test_t = minmaxed[amount_totrain:amount_totrain+amount_totest,1].tolist()
z_test_t = minmaxed[amount_totrain:amount_totrain+amount_totest,2].tolist()
x_test_r = minmaxed[1+amount_totrain:amount_totrain+amount_totest+1,0].tolist()
y_test_r = minmaxed[1+amount_totrain:amount_totrain+amount_totest+1,1].tolist()
z_test_r = minmaxed[1+amount_totrain:amount_totrain+amount_totest+1,2].tolist()
dfxtest =  list(map(lambda xt, yt, zt, xr,yr,zr:[(xt,yt,zt),[xr,yr,zr]], x_test_t,y_test_t, z_test_t, x_test_r,y_test_r,z_test_r))

xground = minmaxed[1:amount_totrain+amount_totest+1,0].tolist()
yground = minmaxed[1:amount_totrain+amount_totest+1,1].tolist()
zground = minmaxed[1:amount_totrain+amount_totest+1,2].tolist()

# #df = pd.read_csv("EKG_Reading_Up16_Shift2.csv")

# df1sub1 = point_cloud[:amount_totrain, 1].values.tolist()
# df1sub2 = point_cloud[1:amount_totrain+1, 1].values.tolist()
# df1sub3 = point_cloud[2:amount_totrain+2, 1].values.tolist()
# dfx =  list(map(lambda x, y, z:[(x,y),[z]], df1sub2, df1sub3, df1sub1))
# df2sub1 = point_cloud[:amount_totrain, 1].values.tolist()
# df2sub2 = point_cloud[1:amount_totrain+1, 1].values.tolist()
# df2sub3 = point_cloud[2:amount_totrain+2, 1].values.tolist()
# dfxtest =  list(map(lambda x, y, z:[(x,y),[z]], df2sub2, df2sub3, df2sub1))


nn = NeuralNetwork(learning_rate=0.1, debug=True)
nn.add_layer(n_inputs=3, n_neurons=6)
nn.add_layer(n_inputs=6, n_neurons=3)
nn.train(dataset=dfx, n_iterations=100, print_error_report=True)

# test
actual_outputsx = []
actual_outputsy = []
actual_outputsz = []
actual_outputs = []
for j, (inputs, targets) in enumerate(dfx+dfxtest):
     іses = nn.feed_forward(inputs)
     actual_outputsx.append(іses[0])
     actual_outputsy.append(іses[1])
     actual_outputsz.append(іses[2])
     actual_outputs.append(іses)

#plot3Dref(actual_outputs)
minmaxedNN =minmax_scaler.inverse_transform(actual_outputs)
plot3Dref(minmaxedNN)
# actual_outputsTransx= minmax_scaler.inverse_transform(np.array(actual_outputsx).reshape(-1,1)).tolist()
# flat_listx = [item for sublist in actual_outputsTransx for item in sublist]
# actual_outputsTransy= minmax_scaler.inverse_transform(np.array(actual_outputsy).reshape(-1,1)).tolist()
# flat_listy = [item for sublist in actual_outputsTransy for item in sublist]
# actual_outputsTransz= minmax_scaler.inverse_transform(np.array(actual_outputsz).reshape(-1,1)).tolist()
# flat_listz = [item for sublist in actual_outputsTransz for item in sublist]

fig = go.Figure()

fig.add_trace(
     go.Scatter(y=xground, name='GroundX')
)
fig.add_trace(
     go.Scatter(y=yground, name='GroundY')
)
fig.add_trace(
     go.Scatter(y=zground, name='GroundZ')
)
fig.add_trace(
     go.Scatter(y=actual_outputsx, name='ActualX')
)
fig.add_trace(
     go.Scatter(y=actual_outputsy, name='ActualY')
)
fig.add_trace(
     go.Scatter(y=actual_outputsz, name='ActualZ')
)
fig.show()
#fig2 = px.line(title='ground')
#fig2.add_scatter(x=np.linspace(start_sample , end_sample, totalamount)[3:amount_totrain+amount_totest+3],y=xground, name='XComp')
#fig2.show()

#fig3 = px.line(title='Actual')
#fig3.add_scatter(x=np.linspace(start_sample , end_sample, totalamount)[3:amount_totrain+amount_totest+3],y=flat_list, name='XComp')
#fig3.show()