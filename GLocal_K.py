# Usar python 3.7.13
# Crear un entorno virtual con esa version de python
# Instalar los requerimientos desde el archivo requirements.txt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
from time import time
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
#import random
import os

# Definir la semilla
#seed = 42
#os.environ['PYTHONHASHSEED'] = str(seed)
#np.random.seed(seed)
#random.seed(seed)
#tf.compat.v1.set_random_seed(seed) 

#wandb.login(key="e50d0e9a07181841d024010871ef729807f5d970")
# Ocultar advertencias
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### Data Loader Function

def load_data_framingham(path='./',
                         train_file='Train_Matrix_Clinical_Final.xlsx',
                         test_file='Test_Matrix_Clinical_Final.xlsx',
                         train_mask_file='mask_train_Final.xlsx',
                         test_mask_file='mask_test_Final.xlsx'):

    # Construir rutas absolutas
    train_path = os.path.join(path, train_file)
    test_path = os.path.join(path, test_file)
    train_mask_path = os.path.join(path, train_mask_file)
    test_mask_path = os.path.join(path, test_mask_file)

    # Verificar existencia
    for f in [train_path, test_path, train_mask_path, test_mask_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"No se encontró el archivo: {f}")

    # Cargar archivos (sin encabezado ni índice)
    train = pd.read_excel(train_path, header=None).to_numpy(dtype=np.float32)
    test = pd.read_excel(test_path, header=None).to_numpy(dtype=np.float32)
    train_m = pd.read_excel(train_mask_path, header=None).to_numpy(dtype=np.float32)
    test_m = pd.read_excel(test_mask_path, header=None).to_numpy(dtype=np.float32)

    # Transponer a formato [variables x pacientes]
    train_r = train.T
    test_r = test.T
    train_m = train_m.T
    test_m = test_m.T

    # Verificación de dimensiones
    assert train_r.shape == train_m.shape, "Dimensiones incompatibles entre train y su máscara"
    assert test_r.shape == test_m.shape, "Dimensiones incompatibles entre test y su máscara"

    n_m, n_u = train_r.shape
    
    print(f"Número de variables (n_m): {n_m}")
    print(f"Número de pacientes (n_u): {n_u}")
    print(f"Tamaño matriz train_r: {train_r.shape}")
    print(f"Tamaño matriz test_r: {test_r.shape}")
    print("Tamaño del conjunto de prueba: ",test_r.shape)
    print("Tamaño de la mascara de prueba: ",test_m.shape)

    return n_m, n_u, train_r, train_m, test_r, test_m

def load_data_100k(path='./', delimiter='\t'):
    base_file = os.path.join(path, 'movielens_100k_u1.base')
    test_file = os.path.join(path, 'movielens_100k_u1.test')

    if not os.path.exists(base_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Los archivos no se encontraron en: {path}")

    train = np.loadtxt(base_file, skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(test_file, skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:, 0]).size  # número de usuarios
    n_m = np.unique(total[:, 1]).size  # número de películas
    n_train = train.shape[0]
    n_test = test.shape[0]
    print(n_train)
    print(n_test)

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i, 1] - 1, train[i, 0] - 1] = train[i, 2]

    for i in range(n_test):
        test_r[test[i, 1] - 1, test[i, 0] - 1] = test[i, 2]

    train_m = np.greater(train_r, 1e-12).astype('float32')
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('Data matrix loaded')
    print(f'Number of users: {n_u}')
    print(f'Number of movies: {n_m}')
    print(f'Number of training ratings: {n_train}')
    print(f'Number of test ratings: {n_test}')

    return n_m, n_u, train_r, train_m, test_r, test_m


def load_data_1m(path='./', delimiter='::', frac=0.1, seed=1234):
    file_path = os.path.join(path, 'movielens_1m_dataset.dat')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo movielens_1m_dataset.dat no se encontró en: {path}")

    tic = time()
    print('Reading data...')
    data = np.loadtxt(file_path, skiprows=0, delimiter=delimiter).astype('int32')
    print('Taken', time() - tic, 'seconds')

    n_u = np.unique(data[:, 0]).size
    n_m = np.unique(data[:, 1]).size
    n_r = data.shape[0]

    udict = {u: i for i, u in enumerate(np.unique(data[:, 0]))}
    mdict = {m: i for i, m in enumerate(np.unique(data[:, 1]))}

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        if i < int(frac * n_r):
            test_r[mdict[m_id], udict[u_id]] = r
        else:
            train_r[mdict[m_id], udict[u_id]] = r

    train_m = np.greater(train_r, 1e-12).astype('float32')
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('Data matrix loaded')
    print(f'Number of users: {n_u}')
    print(f'Number of movies: {n_m}')
    print(f'Number of training ratings: {n_r - int(frac * n_r)}')
    print(f'Number of test ratings: {int(frac * n_r)}')

    return n_m, n_u, train_r, train_m, test_r, test_m


def load_matlab_file(path_file, name_field):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = csc_matrix((data, ir, jc)).astype(np.float32)
        else:
            out = np.asarray(ds).astype(np.float32).T
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()
    return out


def load_data_monti(path='./'):
    mat_file = path + 'douban_monti_dataset.mat'

    M = load_matlab_file(mat_file, 'M')
    Otraining = load_matlab_file(mat_file, 'Otraining') * M
    Otest = load_matlab_file(mat_file, 'Otest') * M

    n_u = M.shape[0]  # número de usuarios
    n_m = M.shape[1]  # número de películas
    n_train = Otraining[np.where(Otraining)].size  # cantidad de ratings de entrenamiento
    n_test = Otest[np.where(Otest)].size  # cantidad de ratings de test

    train_r = Otraining.T
    test_r = Otest.T

    train_m = np.greater(train_r, 1e-12).astype('float32')
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('Data matrix loaded')
    print(f'Number of users: {n_u}')
    print(f'Number of movies: {n_m}')
    print(f'Number of training ratings: {n_train}')
    print(f'Number of test ratings: {n_test}')

    return n_m, n_u, train_r, train_m, test_r, test_m


### Load Data

data_path = './data'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
     

# Select a dataset among 'ML-1M', 'ML-100K', and 'Douban'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
dataset = 'Framingham'  # Cambia el dataset activo aquí

try:
    if dataset == 'ML-100K':
        path = data_path + '/MovieLens_100K/'
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k(path=path, delimiter='\t')

    elif dataset == 'ML-1M':
        path = data_path + '/MovieLens_1M/'
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_1m(path=path, delimiter='::', frac=0.1, seed=1234)

    elif dataset == 'Douban':
        path = data_path + '/Douban_monti/'
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_monti(path=path)

    elif dataset == 'Framingham':
        path = './'  # Cambia si tu ruta es diferente
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_framingham(
            path=path,
            train_file= 'Train_MinMax_E3.xlsx',
            test_file=  'Test_MinMax_E1.xlsx',
            train_mask_file= 'mask_train_MinMax_E3.xlsx',
            test_mask_file= 'mask_test_MinMax_E1.xlsx'
        )

    else:
        raise ValueError

except ValueError:
    print('Error: Unable to load data')


### Hyperparameter Settings

n_hid = 500
n_dim = 5
n_layers = 2
gk_size = 3
     

# Different hyperparameter settings for each dataset
                     
if dataset == 'Framingham':
    lambda_2 = 1.0  # l2 regularisation
    lambda_s = 0.000001
    iter_p = 5  # optimisation
    iter_f = 10
    epoch_p = 10  # training epoch
    epoch_f = 10
    dot_scale = 2  # scaled dot product
    
if dataset == 'ML-100K':
    lambda_2 = 20.  # l2 regularisation
    lambda_s = 0.006
    iter_p = 5  # optimisation
    iter_f = 5
    epoch_p = 30  # training epoch
    epoch_f = 60
    dot_scale = 1  # scaled dot product

elif dataset == 'ML-1M':
    lambda_2 = 70.
    lambda_s = 0.018
    iter_p = 50
    iter_f = 10
    epoch_p = 20
    epoch_f = 30
    dot_scale = 0.5

elif dataset == 'Douban':
    lambda_2 = 10.
    lambda_s = 0.022
    iter_p = 5
    iter_f = 5
    epoch_p = 20
    epoch_f = 60
    dot_scale = 2

# Guardar archivos de Excel
df_train_r = pd.DataFrame(train_r)
df_train_r.to_excel("train_input_Final_ceros.xlsx", index=False)
df_test_r = pd.DataFrame(test_r)
df_test_r.to_excel("test_input_Final_ceros.xlsx", index=False)

df_train_m = pd.DataFrame(train_m)
df_train_m.to_excel("mask_train_input_Final_ceros.xlsx", index=False)
df_test_m = pd.DataFrame(test_m)
df_test_m.to_excel("mask_test_input_Final_ceros.xlsx", index=False)


R = tf.placeholder("float", [n_m, n_u])


### Network Function

def local_kernel(u, v):

    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist**2)

    return hat
     

def kernel_layer(x, n_hid=n_hid, n_dim=n_dim, activation=tf.nn.sigmoid, lambda_s=lambda_s, lambda_2=lambda_2, name=''):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        n_in = x.get_shape().as_list()[1]
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-3))
        b = tf.get_variable('b', [n_hid])

    w_hat = local_kernel(u, v)
    
    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])
    
    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    W_eff = W * w_hat  # Local kernelised weight matrix
    y = tf.matmul(x, W_eff) + b
    y = activation(y)
    return y, sparse_reg_term + l2_reg_term
     

def global_kernel(input, gk_size, dot_scale):

    avg_pooling = tf.reduce_mean(input, axis=1)  # Item (axis=1) based average pooling
    print("Average_pooling: ",avg_pooling.shape)
    avg_pooling = tf.reshape(avg_pooling, [1, -1])
    print("Average_pooling: ",avg_pooling.shape)
    n_kernel = avg_pooling.shape[1].value
    print("n_kernel: ",n_kernel)

    conv_kernel = tf.get_variable('conv_kernel', initializer=tf.random.truncated_normal([n_kernel, gk_size**2], stddev=0.1))
    gk = tf.matmul(avg_pooling, conv_kernel) * dot_scale  # Scaled dot product
    gk = tf.reshape(gk, [gk_size, gk_size, 1, 1])

    return gk
     

def global_conv(input, W):

    input = tf.reshape(input, [1, input.shape[0], input.shape[1], 1])
    conv2d = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME'))

    return tf.reshape(conv2d, [conv2d.shape[1], conv2d.shape[2]])


### Network Instantiation

#### Pre-training
print('.-^-._' * 12)
print("PRE-TRAINING")
print("Input:", R.shape)
y = R
reg_losses = None

for i in range(n_layers):
    y, reg_loss = kernel_layer(y, name=str(i))
    print(f" Layer {i}, shape: {y.shape}")
    reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

pred_p, reg_loss = kernel_layer(y, n_u, activation=tf.identity, name='out')
print("Output:", pred_p.shape)
reg_losses = reg_losses + reg_loss

# L2 loss
diff = train_m * (train_r - pred_p)
sqE = tf.nn.l2_loss(diff)
loss_p = sqE + reg_losses
optimizer_p = tf.contrib.opt.ScipyOptimizerInterface(loss_p, options={'disp': True, 'maxiter': iter_p, 'maxcor': 10}, method='L-BFGS-B')


#### Fine-tuning
print('.-^-._' * 12)
print("FINE-TUNING")
print("Input:", R.shape)
y = R
reg_losses = None

for i in range(n_layers):
    y, _ = kernel_layer(y, name=str(i))
    print(f" Layer {i}, shape: {y.shape}")

y_dash, _ = kernel_layer(y, n_u, activation=tf.identity, name='out')
print("Output:", y_dash.shape)

gk = global_kernel(y_dash, gk_size, dot_scale)  # Global kernel
y_hat = global_conv(train_r, gk)  # Global kernel-based rating matrix

for i in range(n_layers):
    y_hat, reg_loss = kernel_layer(y_hat, name=str(i))
    print(f"After layer {i}, shape: {y_hat.shape}")
    reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

pred_f, reg_loss = kernel_layer(y_hat, n_u, activation=tf.identity, name='out')
print("Final output shape:", y.shape)
reg_losses = reg_losses + reg_loss

# L2 loss
diff = train_m * (train_r - pred_f)
sqE = tf.nn.l2_loss(diff)
loss_f = sqE + reg_losses

optimizer_f = tf.contrib.opt.ScipyOptimizerInterface(loss_f, options={'disp': True, 'maxiter': iter_f, 'maxcor': 10}, method='L-BFGS-B')


#### Evaluation code

def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg
     

def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm
     

def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num

#wandb.init(
#    project="GLocalK-framingham",
#    name="glocalk_Comparable_Final_ceros",
#    config={
#        "epochs_pretrain": epoch_p,
#        "epochs_finetune": epoch_f,
#        "scaler": "RobustScaler",
#        "dataset": "Framingham",
#        "layers": n_layers,
#        "train_patients": train_r.shape[1],
#        "test_patients": test_r.shape[1]
#    }
#)

os.makedirs("model_checkpoints", exist_ok=True)
os.makedirs("Cerotocero5", exist_ok=True)
saver = tf.train.Saver()

### Training and Test Loop
best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0
train_rmse_hist, test_rmse_hist, train_pre, test_pre = [], [], [], []
train_mae_hist, test_mae_hist = [], []
train_ndcg_hist, test_ndcg_hist = [], []

time_cumulative = 0
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch_p):
        tic = time()
        optimizer_p.minimize(sess, feed_dict={R: train_r})
        pre = sess.run(pred_p, feed_dict={R: train_r})
        t = time() - tic
        time_cumulative += t
        
        #error = (test_m * (np.clip(pre, 0.00, 1.00) - test_r) ** 2).sum() / test_m.sum()  # test error
        #test_rmse = np.sqrt(error)

        error_train = (train_m * (np.clip(pre, 0.00, 1.00) - train_r) ** 2).sum() / train_m.sum()
        train_rmse = np.sqrt(error_train)
        
        train_pre.append(train_rmse)
        #test_pre.append(test_rmse)
        
        #wandb.log({
        #"epoch": i + 1,
        #"phase": "pretrain",
        #"train_rmse": train_rmse,
        #"test_rmse": test_rmse
        #})

        print('.-^-._' * 12)
        print('PRE-TRAINING')
        print('Epoch:', i+1, 'train rmse:', train_rmse)
        print('Time:', t, 'seconds')
        print('Time cumulative:', time_cumulative, 'seconds')
        print('.-^-._' * 12)

    pred_pre = pd.DataFrame(pre)
    pred_pre.to_excel(f"Cerotocero5/pred_pre_{lambda_2}vs{lambda_s}.xlsx", index=False, header=False)
    
    #val_pred = pd.DataFrame(test_r)
    #val_pred.to_excel("Lambda/val_r.xlsx", index=False, header=False)

    train_pred = pd.DataFrame(train_r)
    df_real = train_pred.iloc[12:17, 9301:11627] 
    train_pred.to_excel("Cerotocero5/train_r.xlsx", index=False, header=False)

    for i in range(epoch_f):
        tic = time()
        optimizer_f.minimize(sess, feed_dict={R: train_r})
        pre = sess.run(pred_f, feed_dict={R: train_r})

        t = time() - tic
        time_cumulative += t
        #error = (test_m * (np.clip(pre, 0.00, 1.00) - test_r) ** 2).sum() / test_m.sum()  # test error
        #test_rmse = np.sqrt(error)

        error_train = (train_m * (np.clip(pre, 0.00, 1.00) - train_r) ** 2).sum() / train_m.sum()  # train error
        train_rmse = np.sqrt(error_train)

        #test_mae = (test_m * np.abs(np.clip(pre, 0.00, 1.00) - test_r)).sum() / test_m.sum()
        train_mae = (train_m * np.abs(np.clip(pre, 0.00, 1.00) - train_r)).sum() / train_m.sum()
        
        #test_ndcg = call_ndcg(np.clip(pre, 0.00, 1.00), test_r)
        train_ndcg = call_ndcg(np.clip(pre, 0.00, 1.00), train_r)

        train_rmse_hist.append(train_rmse)
        #test_rmse_hist.append(test_rmse)

        train_mae_hist.append(train_mae)
        #test_mae_hist.append(test_mae)

        train_ndcg_hist.append(train_ndcg)
        #test_ndcg_hist.append(test_ndcg)

        if train_rmse < best_rmse:
            best_rmse = train_rmse
            best_rmse_ep = i+1
            saver.save(sess, "model_checkpoints/LGlobalK_best_rmse_val.ckpt")

        if train_mae < best_mae:
            best_mae = train_mae
            best_mae_ep = i+1

        if best_ndcg < train_ndcg:
            best_ndcg = train_ndcg
            best_ndcg_ep = i+1
        
        #wandb.log({
        #"epoch": i + 1,
        #"phase": "finetune",
        #"train_rmse": train_rmse,
        #"test_rmse": test_rmse,
        #"train_mae": train_mae,
        #"test_mae": test_mae,
        #"train_ndcg": train_ndcg,
        #"test_ndcg": test_ndcg
        #})
        
        print('.-^-._' * 12)
        print('FINE-TUNING')
        #print('Epoch:', i+1, 'val rmse:', test_rmse, 'val mae:', test_mae, 'val ndcg:', test_ndcg)
        print('Epoch:', i+1, 'train rmse:', train_rmse, 'train mae:', train_mae, 'train ndcg:', train_ndcg)
        print('Time:', t, 'seconds')
        print('Time cumulative:', time_cumulative, 'seconds')
        print('.-^-._' * 12)

    pred_fine = pd.DataFrame(pre)
    df_pred = pred_fine.iloc[12:17, 9301:11627]
    pred_fine.to_excel(f"Cerotocero5/pred_fine_{lambda_2}vs{lambda_s}.xlsx", index=False, header=False)
    

        # Realizar la predicción
    #pre_final = sess.run(pred_f, feed_dict={R: train_r})
    #df_pred_f = pd.DataFrame(pre_final)
    #df_pred_f.to_excel("predicted_output_Final_ceros.xlsx", index=False)

errors = df_real.values - df_pred.values
rmse_yhat_test = np.sqrt(np.mean(errors ** 2))
mae_yhat_test  = np.mean(np.abs(errors))

# Final result
print('Epoch:', best_rmse_ep, ' best rmse:', best_rmse)
print('Epoch:', best_mae_ep, ' best mae:', best_mae)
print('Epoch:', best_ndcg_ep, ' best ndcg:', best_ndcg)

filename = f"Cerotocero5/resultados_{lambda_2}vs{lambda_s}.txt"

with open(filename, 'w') as f:
    f.write(f"Resultados para lambda_2 = {lambda_2}, lambda_s = {lambda_s}\n")
    f.write(f"Epoch: {best_rmse_ep}  best rmse: {best_rmse:.6f}\n")
    f.write(f"Epoch: {best_mae_ep}  best mae: {best_mae:.6f}\n")
    f.write(f"Epoch: {best_ndcg_ep}  best ndcg: {best_ndcg:.6f}\n")
    f.write(f"RMSE Y^test: {rmse_yhat_test:.6f}\n")
    f.write(f"MAE Y^test: {mae_yhat_test:.6f}\n")

# Preentrenamiento

epochs = list(range(1, epoch_p + 1))

plt.figure(figsize=(18, 5))

# RMSE
plt.plot(epochs, train_pre, label='Train RMSE')
#plt.plot(epochs, test_pre, label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE Evolution')
plt.legend()
plt.savefig(f'Cerotocero5/pre_{lambda_2}vs{lambda_s}.png')
plt.close()
#plt.show()

# Gráficas de desempeño

epochs = list(range(1, epoch_f + 1))

plt.figure(figsize=(18, 5))

# RMSE
plt.subplot(1, 3, 1)
plt.plot(epochs, train_rmse_hist, label='Train RMSE')
#plt.plot(epochs, test_rmse_hist, label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE Evolution')
plt.legend()

# MAE
plt.subplot(1, 3, 2)
plt.plot(epochs, train_mae_hist, label='Train MAE')
#plt.plot(epochs, test_mae_hist, label='Test MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE Evolution')
plt.legend()

# NDCG
plt.subplot(1, 3, 3)
plt.plot(epochs, train_ndcg_hist, label='Train NDCG')
#plt.plot(epochs, test_ndcg_hist, label='Test NDCG')
plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.title('NDCG Evolution')
plt.legend()

plt.tight_layout()
plt.savefig(f'Cerotocero5/fine_{lambda_2}vs{lambda_s}.png')
plt.close()
#plt.show()