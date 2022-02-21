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

amount_totrain = 1000
amount_totest = 500
start_sample = 0
end_sample = 3
sample_rate = 5000
start_vec = [0,0,0.0002]
totalamount = amount_totrain+amount_totest + 5 

#sinewave = samplefuntion2d(func_complex, start_sample , end_sample , totalamount)
chell_sample = func_ode_sample(start_sample,end_sample,sample_rate,start_vec[0],start_vec[1],start_vec[2])
plot3Dref(chell_sample)
minmax_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
data_minmaxX = minmax_scaler.fit_transform(chell_sample[:amount_totrain+amount_totest+3,0].reshape(-1,1))
data_minmaxY = minmax_scaler.fit_transform(chell_sample[:amount_totrain+amount_totest+3,1].reshape(-1,1))
data_minmaxZ = minmax_scaler.fit_transform(chell_sample[:amount_totrain+amount_totest+3,2].reshape(-1,1))

x_train = data_minmaxX[:amount_totrain].tolist()
y_train = data_minmaxY[:amount_totrain].tolist()
z_train = data_minmaxZ[:amount_totrain].tolist()
x_recive = data_minmaxX[1:amount_totrain+1].tolist()
y_recive = data_minmaxY[1:amount_totrain+1].tolist()
z_recive = data_minmaxZ[1:amount_totrain+1].tolist()
dfx =  list(map(lambda xt, yt, zt, xr,yr,zr:[(xt[0],yt[0],zt[0]),[xr[0],yr[0],zr[0]]], x_train,y_train, z_train, x_recive,y_recive,z_recive))

x_test_t = data_minmaxX[:amount_totrain].tolist()
y_test_t = data_minmaxY[:amount_totrain].tolist()
z_test_t = data_minmaxZ[:amount_totrain].tolist()
x_test_r = data_minmaxX[1:amount_totrain+1].tolist()
y_test_r = data_minmaxY[1:amount_totrain+1].tolist()
z_test_r = data_minmaxZ[1:amount_totrain+1].tolist()
dfxtest =  list(map(lambda xt, yt, zt, xr,yr,zr:[(xt[0],yt[0],zt[0]),[xr[0],yr[0],zr[0]]], x_test_t,y_test_t, z_test_t, x_test_r,y_test_r,z_test_r))

xground = chell_sample[:amount_totrain+amount_totest+3,0].tolist()
yground = chell_sample[:amount_totrain+amount_totest+3,1].tolist()
zground = chell_sample[:amount_totrain+amount_totest+3,2].tolist()

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
actual_outputsTrans = []
for j, (inputs, targets) in enumerate(dfx+dfxtest):
     іses = nn.feed_forward(inputs)
     actual_outputsx.append(іses[0])
     actual_outputsy.append(іses[1])
     actual_outputsz.append(іses[2])

actual_outputsTransx= minmax_scaler.inverse_transform(np.array(actual_outputsx).reshape(-1,1)).tolist()
flat_listx = [item for sublist in actual_outputsTransx for item in sublist]
actual_outputsTransy= minmax_scaler.inverse_transform(np.array(actual_outputsy).reshape(-1,1)).tolist()
flat_listy = [item for sublist in actual_outputsTransy for item in sublist]
actual_outputsTransz= minmax_scaler.inverse_transform(np.array(actual_outputsz).reshape(-1,1)).tolist()
flat_listz = [item for sublist in actual_outputsTransz for item in sublist]

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
     go.Scatter(y=flat_listx, name='ActualX')
)
fig.add_trace(
     go.Scatter(y=flat_listy, name='ActualY')
)
fig.add_trace(
     go.Scatter(y=flat_listz, name='ActualY')
)
fig.show()
#fig2 = px.line(title='ground')
#fig2.add_scatter(x=np.linspace(start_sample , end_sample, totalamount)[3:amount_totrain+amount_totest+3],y=xground, name='XComp')
#fig2.show()

#fig3 = px.line(title='Actual')
#fig3.add_scatter(x=np.linspace(start_sample , end_sample, totalamount)[3:amount_totrain+amount_totest+3],y=flat_list, name='XComp')
#fig3.show()