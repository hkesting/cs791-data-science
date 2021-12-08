from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, Dense, Dropout, LSTM
from keras.models import Model
from keras.activations import softmax
from keras.losses import categorical_crossentropy

from keras.layers.core import Dense, Dropout, Activation, Masking

from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import mean_squared_error

# Simulate state 1 to 2


for sim in range(5):
    sim1 = data_std.loc[idx_sim7]
    for r in range(2704, 3004, 1):
        # Let's start with sim1
        sim_step_in = sim1.iloc[0:(r + 1), np.r_[10:173]]
        sim_step_in = scaler_in.transform(sim_step_in)
        sim_step_in = sim_step_in.reshape(1, (r + 1), 163)
        predstate = model_class.predict(sim_step_in)
        predstate = predstate[-1]
        predstate = predstate[-1]
        predstate = predstate / np.sum(predstate)
        sim1['state'][r] = getClass(predstate)
        if r >= 999:
            sim_step_in = sim_step_in[0, (r - 999):(r + 1), :]
            sim_step_in = sim_step_in.reshape(1, 1000, 163)
        if sim1['state'][r] == 1:
            vxvy_invtransform = scaler_out.inverse_transform(model1.predict(sim_step_in))
        elif sim1['state'][r] == 2:
            vxvy_invtransform = scaler_out.inverse_transform(model2.predict(sim_step_in))
        elif sim1['state'][r] == 4:
            vxvy_invtransform = scaler_out.inverse_transform(model4.predict(sim_step_in))
        else:
            vxvy_invtransform = scaler_out.inverse_transform(model5.predict(sim_step_in))
        sim1.iloc[r:(r + 1), np.r_[7:9]] = vxvy_invtransform[-1]
        sim1['x'][r:(r + 1)] = sim1['x-1'][r:(r + 1)] + sim1['vx'][r:(r + 1)]
        sim1['y'][r:(r + 1)] = sim1['y-1'][r:(r + 1)] + sim1['vy'][r:(r + 1)]
        sim1['x-1'][(r + 1):(r + 2)] = sim1['x'][r:(r + 1)]
        sim1['y-1'][(r + 1):(r + 2)] = sim1['y'][r:(r + 1)]
        sim1['vx-1'][(r + 1):(r + 2)] = sim1['vx'][r:(r + 1)]
        sim1['vy-1'][(r + 1):(r + 2)] = sim1['vy'][r:(r + 1)]
        # update lagged x,y
        sim1.iloc[(r + 1):(r + 2), np.r_[13:52]] = sim1.iloc[r:(r + 1), np.r_[12:51]].values  # update x-2:x-40
        sim1.iloc[(r + 1):(r + 2), 53:92] = sim1.iloc[r:(r + 1), np.r_[52:91]].values  # update y-2:y-40
        # update lagged vx, vy
        sim1.iloc[(r + 1):(r + 2), 93:132] = sim1.iloc[r:(r + 1), np.r_[92:131]].values  # update vx-2:vx-40
        sim1.iloc[(r + 1):(r + 2), 133:172] = sim1.iloc[r:(r + 1), np.r_[132:171]].values  # update vy-2:vy-40
        # update dist
        sim1['dist'][(r + 1):(r + 2)] = veuc_dist(sim1['x-1'][(r + 1):(r + 2)], sim1['x-2'][(r + 1):(r + 2)],
                                                  sim1['y-1'][(r + 1):(r + 2)], sim1['y-2'][(r + 1):(r + 2)])
    get_sim = sim1.to_numpy()
    filename = '/LSTM/' + str(sim) + 'LSTM7_1_2.npy'
    np.save(filename, get_sim)

# Simulate state 4 to 5


for sim in range(5):
    sim1 = data_std.loc[idx_sim7]
    for r in range(13772, 14072, 1):
        # Let's start with sim1
        sim_step_in = sim1.iloc[0:(r + 1), np.r_[10:173]]
        sim_step_in = scaler_in.transform(sim_step_in)
        sim_step_in = sim_step_in.reshape(1, (r + 1), 163)
        predstate = model_class.predict(sim_step_in)
        predstate = predstate[-1]
        predstate = predstate[-1]
        predstate = predstate / np.sum(predstate)
        sim1['state'][r] = getClass(predstate)
        if r >= 999:
            sim_step_in = sim_step_in[0, (r - 999):(r + 1), :]
            sim_step_in = sim_step_in.reshape(1, 1000, 163)
        if sim1['state'][r] == 1:
            vxvy_invtransform = scaler_out.inverse_transform(model1.predict(sim_step_in))
        elif sim1['state'][r] == 2:
            vxvy_invtransform = scaler_out.inverse_transform(model2.predict(sim_step_in))
        elif sim1['state'][r] == 4:
            vxvy_invtransform = scaler_out.inverse_transform(model4.predict(sim_step_in))
        else:
            vxvy_invtransform = scaler_out.inverse_transform(model5.predict(sim_step_in))
        sim1.iloc[r:(r + 1), np.r_[7:9]] = vxvy_invtransform[-1]
        sim1['x'][r:(r + 1)] = sim1['x-1'][r:(r + 1)] + sim1['vx'][r:(r + 1)]
        sim1['y'][r:(r + 1)] = sim1['y-1'][r:(r + 1)] + sim1['vy'][r:(r + 1)]
        sim1['x-1'][(r + 1):(r + 2)] = sim1['x'][r:(r + 1)]
        sim1['y-1'][(r + 1):(r + 2)] = sim1['y'][r:(r + 1)]
        sim1['vx-1'][(r + 1):(r + 2)] = sim1['vx'][r:(r + 1)]
        sim1['vy-1'][(r + 1):(r + 2)] = sim1['vy'][r:(r + 1)]
        # update lagged x,y
        sim1.iloc[(r + 1):(r + 2), np.r_[13:52]] = sim1.iloc[r:(r + 1), np.r_[12:51]].values  # update x-2:x-40
        sim1.iloc[(r + 1):(r + 2), 53:92] = sim1.iloc[r:(r + 1), np.r_[52:91]].values  # update y-2:y-40
        # update lagged vx, vy
        sim1.iloc[(r + 1):(r + 2), 93:132] = sim1.iloc[r:(r + 1), np.r_[92:131]].values  # update vx-2:vx-40
        sim1.iloc[(r + 1):(r + 2), 133:172] = sim1.iloc[r:(r + 1), np.r_[132:171]].values  # update vy-2:vy-40
        # update dist
        sim1['dist'][(r + 1):(r + 2)] = veuc_dist(sim1['x-1'][(r + 1):(r + 2)], sim1['x-2'][(r + 1):(r + 2)],
                                                  sim1['y-1'][(r + 1):(r + 2)], sim1['y-2'][(r + 1):(r + 2)])
    get_sim = sim1.to_numpy()
    filename = '/LSTM/' + str(sim) + 'LSTM7_4_5.npy'
    np.save(filename, get_sim)

# Simulate during state 2


for sim in range(5):
    sim1 = data_std.loc[idx_sim7]
    for r in range(2890, 3190, 1):
        # Let's start with sim1
        sim_step_in = sim1.iloc[0:(r + 1), np.r_[10:173]]
        sim_step_in = scaler_in.transform(sim_step_in)
        sim_step_in = sim_step_in.reshape(1, (r + 1), 163)
        predstate = model_class.predict(sim_step_in)
        predstate = predstate[-1]
        predstate = predstate[-1]
        predstate = predstate / np.sum(predstate)
        sim1['state'][r] = getClass(predstate)
        if r >= 999:
            sim_step_in = sim_step_in[0, (r - 999):(r + 1), :]
            sim_step_in = sim_step_in.reshape(1, 1000, 163)
        if sim1['state'][r] == 1:
            vxvy_invtransform = scaler_out.inverse_transform(model1.predict(sim_step_in))
        elif sim1['state'][r] == 2:
            vxvy_invtransform = scaler_out.inverse_transform(model2.predict(sim_step_in))
        elif sim1['state'][r] == 4:
            vxvy_invtransform = scaler_out.inverse_transform(model4.predict(sim_step_in))
        else:
            vxvy_invtransform = scaler_out.inverse_transform(model5.predict(sim_step_in))
        sim1.iloc[r:(r + 1), np.r_[7:9]] = vxvy_invtransform[-1]
        sim1['x'][r:(r + 1)] = sim1['x-1'][r:(r + 1)] + sim1['vx'][r:(r + 1)]
        sim1['y'][r:(r + 1)] = sim1['y-1'][r:(r + 1)] + sim1['vy'][r:(r + 1)]
        sim1['x-1'][(r + 1):(r + 2)] = sim1['x'][r:(r + 1)]
        sim1['y-1'][(r + 1):(r + 2)] = sim1['y'][r:(r + 1)]
        sim1['vx-1'][(r + 1):(r + 2)] = sim1['vx'][r:(r + 1)]
        sim1['vy-1'][(r + 1):(r + 2)] = sim1['vy'][r:(r + 1)]
        # update lagged x,y
        sim1.iloc[(r + 1):(r + 2), np.r_[13:52]] = sim1.iloc[r:(r + 1), np.r_[12:51]].values  # update x-2:x-40
        sim1.iloc[(r + 1):(r + 2), 53:92] = sim1.iloc[r:(r + 1), np.r_[52:91]].values  # update y-2:y-40
        # update lagged vx, vy
        sim1.iloc[(r + 1):(r + 2), 93:132] = sim1.iloc[r:(r + 1), np.r_[92:131]].values  # update vx-2:vx-40
        sim1.iloc[(r + 1):(r + 2), 133:172] = sim1.iloc[r:(r + 1), np.r_[132:171]].values  # update vy-2:vy-40
        # update dist
        sim1['dist'][(r + 1):(r + 2)] = veuc_dist(sim1['x-1'][(r + 1):(r + 2)], sim1['x-2'][(r + 1):(r + 2)],
                                                  sim1['y-1'][(r + 1):(r + 2)], sim1['y-2'][(r + 1):(r + 2)])
    get_sim = sim1.to_numpy()
    filename = '/LSTM/' + str(sim) + 'LSTM7_2.npy'
    np.save(filename, get_sim)

# simulate during state 5


for sim in range(5):
    sim1 = data_std.loc[idx_sim7]
    for r in range(13824, 14124, 1):
        # Let's start with sim1
        sim_step_in = sim1.iloc[0:(r + 1), np.r_[10:173]]
        sim_step_in = scaler_in.transform(sim_step_in)
        sim_step_in = sim_step_in.reshape(1, (r + 1), 163)
        predstate = model_class.predict(sim_step_in)
        predstate = predstate[-1]
        predstate = predstate[-1]
        predstate = predstate / np.sum(predstate)
        sim1['state'][r] = getClass(predstate)
        if r >= 999:
            sim_step_in = sim_step_in[0, (r - 999):(r + 1), :]
            sim_step_in = sim_step_in.reshape(1, 1000, 163)
        if sim1['state'][r] == 1:
            vxvy_invtransform = scaler_out.inverse_transform(model1.predict(sim_step_in))
        elif sim1['state'][r] == 2:
            vxvy_invtransform = scaler_out.inverse_transform(model2.predict(sim_step_in))
        elif sim1['state'][r] == 4:
            vxvy_invtransform = scaler_out.inverse_transform(model4.predict(sim_step_in))
        else:
            vxvy_invtransform = scaler_out.inverse_transform(model5.predict(sim_step_in))
        sim1.iloc[r:(r + 1), np.r_[7:9]] = vxvy_invtransform[-1]
        sim1['x'][r:(r + 1)] = sim1['x-1'][r:(r + 1)] + sim1['vx'][r:(r + 1)]
        sim1['y'][r:(r + 1)] = sim1['y-1'][r:(r + 1)] + sim1['vy'][r:(r + 1)]
        sim1['x-1'][(r + 1):(r + 2)] = sim1['x'][r:(r + 1)]
        sim1['y-1'][(r + 1):(r + 2)] = sim1['y'][r:(r + 1)]
        sim1['vx-1'][(r + 1):(r + 2)] = sim1['vx'][r:(r + 1)]
        sim1['vy-1'][(r + 1):(r + 2)] = sim1['vy'][r:(r + 1)]
        # update lagged x,y
        sim1.iloc[(r + 1):(r + 2), np.r_[13:52]] = sim1.iloc[r:(r + 1), np.r_[12:51]].values  # update x-2:x-40
        sim1.iloc[(r + 1):(r + 2), 53:92] = sim1.iloc[r:(r + 1), np.r_[52:91]].values  # update y-2:y-40
        # update lagged vx, vy
        sim1.iloc[(r + 1):(r + 2), 93:132] = sim1.iloc[r:(r + 1), np.r_[92:131]].values  # update vx-2:vx-40
        sim1.iloc[(r + 1):(r + 2), 133:172] = sim1.iloc[r:(r + 1), np.r_[132:171]].values  # update vy-2:vy-40
        # update dist
        sim1['dist'][(r + 1):(r + 2)] = veuc_dist(sim1['x-1'][(r + 1):(r + 2)], sim1['x-2'][(r + 1):(r + 2)],
                                                  sim1['y-1'][(r + 1):(r + 2)], sim1['y-2'][(r + 1):(r + 2)])
    get_sim = sim1.to_numpy()
    filename = '/LSTM/' + str(sim) + 'LSTM7_5.npy'
    np.save(filename, get_sim)

# simulate from state 2 to 4


for sim in range(5):
    sim1 = data_std.loc[idx_sim7]
    for r in range(3013, 3313, 1):
        # Let's start with sim1
        sim_step_in = sim1.iloc[0:(r + 1), np.r_[10:173]]
        sim_step_in = scaler_in.transform(sim_step_in)
        sim_step_in = sim_step_in.reshape(1, (r + 1), 163)
        predstate = model_class.predict(sim_step_in)
        predstate = predstate[-1]
        predstate = predstate[-1]
        predstate = predstate / np.sum(predstate)
        sim1['state'][r] = getClass(predstate)
        if r >= 999:
            sim_step_in = sim_step_in[0, (r - 999):(r + 1), :]
            sim_step_in = sim_step_in.reshape(1, 1000, 163)
        if sim1['state'][r] == 1:
            vxvy_invtransform = scaler_out.inverse_transform(model1.predict(sim_step_in))
        elif sim1['state'][r] == 2:
            vxvy_invtransform = scaler_out.inverse_transform(model2.predict(sim_step_in))
        elif sim1['state'][r] == 4:
            vxvy_invtransform = scaler_out.inverse_transform(model4.predict(sim_step_in))
        else:
            vxvy_invtransform = scaler_out.inverse_transform(model5.predict(sim_step_in))
        sim1.iloc[r:(r + 1), np.r_[7:9]] = vxvy_invtransform[-1]
        sim1['x'][r:(r + 1)] = sim1['x-1'][r:(r + 1)] + sim1['vx'][r:(r + 1)]
        sim1['y'][r:(r + 1)] = sim1['y-1'][r:(r + 1)] + sim1['vy'][r:(r + 1)]
        sim1['x-1'][(r + 1):(r + 2)] = sim1['x'][r:(r + 1)]
        sim1['y-1'][(r + 1):(r + 2)] = sim1['y'][r:(r + 1)]
        sim1['vx-1'][(r + 1):(r + 2)] = sim1['vx'][r:(r + 1)]
        sim1['vy-1'][(r + 1):(r + 2)] = sim1['vy'][r:(r + 1)]
        # update lagged x,y
        sim1.iloc[(r + 1):(r + 2), np.r_[13:52]] = sim1.iloc[r:(r + 1), np.r_[12:51]].values  # update x-2:x-40
        sim1.iloc[(r + 1):(r + 2), 53:92] = sim1.iloc[r:(r + 1), np.r_[52:91]].values  # update y-2:y-40
        # update lagged vx, vy
        sim1.iloc[(r + 1):(r + 2), 93:132] = sim1.iloc[r:(r + 1), np.r_[92:131]].values  # update vx-2:vx-40
        sim1.iloc[(r + 1):(r + 2), 133:172] = sim1.iloc[r:(r + 1), np.r_[132:171]].values  # update vy-2:vy-40
        # update dist
        sim1['dist'][(r + 1):(r + 2)] = veuc_dist(sim1['x-1'][(r + 1):(r + 2)], sim1['x-2'][(r + 1):(r + 2)],
                                                  sim1['y-1'][(r + 1):(r + 2)], sim1['y-2'][(r + 1):(r + 2)])
    get_sim = sim1.to_numpy()
    filename = '/LSTM/' + str(sim) + 'LSTM7_2_4.npy'
    np.save(filename, get_sim)