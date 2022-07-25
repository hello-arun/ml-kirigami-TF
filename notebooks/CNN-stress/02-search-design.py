import tensorflow as tf
print(tf.__version__)
import matplotlib
# matplotlib.use('TKAgg')
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
random = np.random
random.seed(142536)
PROJECT_DIR = "../../"
import os
os.makedirs("./results",exist_ok=True)
# ## Helper class

class Helper:
    import numpy as np

    def make_fine_grid(self,raw_data, n_corse_x = 3, n_corse_y = 5, n_fine_x = 30, n_fine_y = 80):
        listFG = []  # List Fine Grid
        N = len(raw_data)
        for i in range(N):
            print(f"Pre Processing {i+1:05d}/{N}, {100*(i+1)//N}%",end="\r",flush=True)
            kirigami_config = raw_data[i, 0:15]
            inner_wCuts = self.corse_to_fine_config(
                kirigami_config, n_corse_x, n_corse_y, n_fine_x, n_fine_y)
            listFG.append(inner_wCuts)

        alldata_FG = np.array(listFG)
        alldata_FG = np.append(alldata_FG, raw_data[:, -3:], 1)
        return alldata_FG

    def corse_to_fine_config(self,kirigami_config, n_corse_x, n_corse_y, n_fine_x, n_fine_y):
        """
        Make Fine Grid using corse grid
        0 5 10     0  80  160 ... 2320
        1 6 11     .  .    .  ... .
        2 7 12  => .  .    .  ... .
        3 8 13     .  .    .  ... .
        4 9 14     79 159 239 ... 2399

        Parameters
        --------------------
        kirigami_config: Corse Kirigami config of size n_corse_x * n_corse_y
        return: Fine grid 1D array of size n_fine_x*n_fine_y
        """
        fine_grid = np.ones((n_fine_x,n_fine_y))
        mx, my = n_fine_x//n_corse_x, n_fine_y//n_corse_y  # 10 16
        zeros = np.array([1]*mx)[:,np.newaxis]
        zeros[mx//3:2*mx//3+1]=0
        # ONLY MAKE CUTS inside the INNER REGION !!
        for index,num in enumerate(kirigami_config):
            if num == 0:
                i_corse_x = index // n_corse_y
                i_corse_y = index % n_corse_y
                fine_grid[mx*i_corse_x:mx*(i_corse_x+1),my*i_corse_y:my*(i_corse_y+1)] = zeros
        return fine_grid.flatten()

    def split_data(self,x, y, frac_training):
        """
        Parameters
        ----------------------------
        x: numpy matrix 
        y: numpy array
        percentage_test: float 
            percentage of data to be set for test set
        Return
        -------------------
        X_train, X_valid, X_test, y_train, y_valid, y_test

        """
        frac_valid = 1-frac_training
        ntrain = int(frac_training * len(x))
        nvalid = int(frac_valid * len(x))
        # ntest = int(frac_valid * len(x))

        X_train = x[:ntrain].reshape((-1,30,80,1))
        X_valid = x[ntrain:ntrain+nvalid].reshape((-1,30,80,1))

        y_train = y[:ntrain]
        y_valid = y[ntrain:ntrain+nvalid]
        # [:,np.newaxis] to convert it to column array
        return X_train, X_valid, y_train[:,np.newaxis], y_valid[:,np.newaxis]

helper = Helper()

# ## Prepare Dataset
# We conver coarse grid data to fine grid data using helper class
alldata_15G = np.loadtxt(f'{PROJECT_DIR}/raw/alldata_15G.dat')
alldata_FG  = helper.make_fine_grid(alldata_15G)
print("\nDone..")
# Rescale the data to make it convinent 
i_strain,i_toughness,i_stress=-3,-2,-1
_max_strain = np.max(alldata_FG[:,i_strain])
_max_toughness = np.max(alldata_FG[:,i_toughness])
_max_stress = np.max(alldata_FG[:,i_stress])
print("Rescaling Properties..")
print("Max Strain:",_max_strain)
print("Max Stress:",_max_stress)
print("Max Toughn:",_max_toughness)
alldata_FG[:,i_strain] /= _max_strain
alldata_FG[:,i_stress] /= _max_stress
alldata_FG[:,i_toughness] /= _max_toughness
print("Done..")
# Shuffle the data althought not needed as it is already suffled
np.random.shuffle(alldata_FG)

# ## Basic Setup
num_samples = 100  # Number of initial samples
num_generations = 16 # Numbers of generations to run to find optimal designs
sample_indices = np.random.choice(len(alldata_FG), num_samples, replace=False)
alldata_Train = alldata_FG[sample_indices]
alldata_Random = alldata_FG[sample_indices]
alldata_Random_excluded = np.delete(alldata_FG,sample_indices,axis=0)
alldata_exluded = np.delete(alldata_FG,sample_indices,axis=0)
print(alldata_Train.shape, alldata_exluded.shape)

__FEATURES = len(alldata_Train[0])-3
__PROP_INDEX = i_stress  # -3: Strain, -1: Stress 
sample_indices = np.argpartition(alldata_FG[:,__PROP_INDEX],-num_samples)[-num_samples:]
__AVG_TRUE_TOP = np.average(alldata_FG[:,__PROP_INDEX][sample_indices])
print("No of featues:",__FEATURES,"\nProperty index:",__PROP_INDEX)
I_GEN=1
avg_tops=np.empty(shape=(0,3))
while I_GEN<=num_generations:
    print("Generation:",I_GEN)
    X_train, X_valid, y_train, y_valid = helper.split_data(alldata_Train[:,0:__FEATURES], alldata_Train[:,__PROP_INDEX], 0.9)
    X_test_ex, _, y_test_ex, _         = helper.split_data(alldata_exluded[:,0:__FEATURES], alldata_exluded[:,__PROP_INDEX], 1.0)
    print("Training",len(X_train))
    print("Validation",len(X_valid))
    print("Test Excluded",len(X_test_ex))
    print(X_train.shape,y_train.shape)

    # ## Regression Model

    n_hidden=64 # The best choice for hidden layer
    model = models.Sequential()
    # Filters:16, Kernal:3x3
    model.add(layers.Conv2D(
        input_shape = (30,80,1),
        filters = 16,
        kernel_size = [3,3],
        padding="same",
        activation="relu"))
    # Max Pooling, kernel:2x2, strides:2 
    model.add(layers.MaxPooling2D((2,2),2))
    # Filters:32, Kernal:3x3
    model.add(layers.Conv2D(32,(3,3),activation="relu"))
    # Max Pooling, kernel:2x2, strides:2 
    model.add(layers.MaxPooling2D((2,2),2))
    # Filters:64, Kernal:3x3
    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    # Max Pooling, kernel:2x2, strides:2 
    model.add(layers.MaxPooling2D((2,2),2))

    # Flatten for the fully connected layer
    model.add(layers.Flatten())
    # Fully connected layer 64:Neuron
    model.add(layers.Dense(n_hidden,activation="relu"))
    model.add(layers.Dense(1,activation="linear"))

    model.summary()

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error'])

    history = model.fit(X_train, y_train, batch_size=num_samples//10,epochs=30, 
                        validation_data=(X_valid, y_valid))
    model.save(f"./models/02-search/V1.0/Gen-{I_GEN}/f16-f32-f64-h{n_hidden}")


    # Plot progress
    sample_indices = np.argpartition(alldata_Train[:,__PROP_INDEX],-num_samples)[-num_samples:]
    avg_top = np.average(alldata_Train[:,__PROP_INDEX][sample_indices])

    sample_indices = np.argpartition(alldata_Random[:,__PROP_INDEX],-num_samples)[-num_samples:]
    avg_top_random = np.average(alldata_Random[:,__PROP_INDEX][sample_indices])

    avg_tops = np.append(avg_tops,[[I_GEN,avg_top,avg_top_random]],axis=0)
    np.savetxt("./results/02-searched-design.dat",avg_tops,header=f"Generation Avg_Top{num_samples}_Model Avg_Top{num_samples}_Random")
    plt.plot(avg_tops[:,0],avg_tops[:,1],".-",label="Model")
    plt.plot(avg_tops[:,0],avg_tops[:,2],".-",label="Random")
    plt.plot([0,num_samples],[__AVG_TRUE_TOP,__AVG_TRUE_TOP],"--",label="True Avg")
    plt.xlim([1,num_generations])
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(f"./results/02-searched-design.png")
    plt.cla()

    # Move to next Generation
    I_GEN += 1
    # Find new 100 configs for next generation 
    y_pred = model.predict(X_test_ex).flatten() 
    sample_indices = np.argpartition(y_pred,-num_samples)[-num_samples:]  # Best Performers
    alldata_Train = np.append(alldata_Train,alldata_exluded[sample_indices],axis=0)
    np.random.shuffle(alldata_Train)
    alldata_exluded = np.delete(alldata_exluded,sample_indices,axis=0)

    # Find new 100 configs for Random new genearion
    sample_indices  = np.random.choice(len(alldata_Random_excluded), num_samples, replace=False)
    alldata_Random  = np.append(alldata_Random,alldata_Random_excluded[sample_indices],axis=0)
    alldata_Random_excluded = np.delete(alldata_Random_excluded,sample_indices,axis=0)
    
    




# Now remodel with this dataset and continue

# mse = tf.keras.metrics.mean_squared_error(
#     y_test, model.predict(X_test)
# )
# print(np.mean(mse))
# y_pred = model.predict(X_test)
# xx = y_pred.flatten()
# yy = y_test.flatten()
# #plt.plot([0,1],[1,0])

# y_pred = model.predict(X_test)
# xx = y_pred.flatten()
# yy = y_test.flatten()
# fig,(ax1,ax2,ax3)=plt.subplots(1,3)
# fig.set_size_inches(12,4)
# ax1.plot(y_train.flatten(),model.predict(X_train).flatten(),".",label="Training")
# ax2.plot(y_valid.flatten(),model.predict(X_valid).flatten(),".",label="Validation")
# ax3.plot(y_test.flatten(),model.predict(X_test).flatten(),".",label="Testing")

# for ax in (ax1,ax2,ax3):
#     ax.plot([0,1],[0,1],"r--",lw=2.0)
#     ax.legend()
#     ax.set_xlim([0,1])
#     ax.set_ylim([0,1])
# plt.tight_layout()
# plt.show()