import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# Read the CSV files
cols = ['user_id','song_id','weight']
data = pd.read_csv('user_song.csv',names=cols,header=0)
test_data = pd.read_csv('test.csv')

def als_step(data,fixed,k,factor):
    P = fixed.T.dot(fixed) + np.identity(k)*factor
    Q = data.dot(fixed)
    P_inv = np.linalg.inv(P)
    return Q.dot(P_inv)

def find_sse(true,pred):
    return (true - pred)**2

user_ids = data["user_id"].unique()
song_ids = data["song_id"].unique()
user_index = {index: user_id for user_id,index in enumerate(user_ids)}
song_index = {index: song_id for song_id,index in enumerate(song_ids)}
weight_mat = np.random.random((len(user_ids),len(song_ids)))
user_num = len(user_ids)
song_num = len(song_ids)
r_avg = 0
non_zero_count = 0
for row in data.itertuples(index=False):
    weight = row[2]
    if weight != 0:
        weight_mat[user_index[row[0]],song_index[row[1]]] = weight
        r_avg += weight
        non_zero_count +=1
r_avg = r_avg/non_zero_count

# Normalize the weights
for i in range(len(user_ids)):
    for j in range(len(song_ids)):
        if weight_mat[i, j] != 0:
            weight_mat[i, j] -= r_avg

k = 20
factor = 60
user_mat = np.random.random((len(user_ids),k))
song_mat = np.zeros((len(song_ids),k))
new_sse = np.inf
sse = 0
while new_sse - sse > 300000:
    new_sse = sse
    sse = 0
    for i in song_ids:
        users = np.nonzero(weight_mat[:,song_index[i]])[0]
        if len(users) == 0:
            continue
        relevent_users = user_mat[users,:]
        song_mat[song_index[i],:] = np.linalg.solve(np.dot(relevent_users.T,relevent_users) + factor*np.identity(k),
                                        np.dot(relevent_users.T,weight_mat[users,song_index[i]].T))
    for u in user_ids:
        songs = np.nonzero(weight_mat[user_index[u],:])[0]
        if len(songs) == 0:
            continue
        relevent_songs = song_mat[songs,:]
        user_mat[user_index[u],:] = np.linalg.solve(np.dot(relevent_songs.T,relevent_songs) + factor*np.identity(k),
                                        np.dot(relevent_songs.T,weight_mat[user_index[u],songs].T))
    for i in range(data.shape[0]):
        user = data.loc[i, ['user_id']][0]
        song = data.loc[i, ['song_id']][0]
        sse += find_sse(data.loc[i,['weight']][0],
                        np.dot(user_mat[user_index[user]].T, song_mat[song_index[song]]))
test_user = sorted(set(test_data["user_id"].unique()))
test_song = sorted(set(test_data["song_id"].unique()))
test_data["weight"] = [0]*test_data.shape[0]
for i in range(test_data.shape[0]):
    user = test_data.loc[i,['user_id']][0]
    song = test_data.loc[i,['song_id']][0]
    test_data.loc[i,'weight'] = max(np.dot(user_mat[user_index[user]],song_mat[song_index[song]].T), 0)
print(f"result:{sse}")
test_data.to_csv('205736879_ID2_2.csv',index = False)


