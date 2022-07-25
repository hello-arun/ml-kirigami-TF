"""
This notebook converts the coarse-grained structure (3x5) to fine structure (30x80).
"""
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

def makeFineGrid(kirigami_config, n_corse_x, n_corse_y, n_fine_x, n_fine_y):
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
    print(zeros.shape)
    zeros[mx//3:2*mx//3+1]=0

    # ONLY MAKE CUTS inside the INNER REGION !!
    for index,num in enumerate(kirigami_config):
        if num == 0:
            i_corse_x = index // n_corse_y
            i_corse_y = index % n_corse_y
            fine_grid[mx*i_corse_x:mx*(i_corse_x+1),my*i_corse_y:my*(i_corse_y+1)] = zeros
    return fine_grid.flatten()

def create_matrix(data, discrete, prop, cutoff, nfeatures):
    """
    Create matrix X and 1D array y for data analysis 

    Parameters
    -----------------
    data: numpy array
        data  
    discrete: boolean
        If TRUE set y=1 for 'good' design and y=0 for 'bad' design
    prop: int 
        property to study
    cutoff: float 
        Set cutoff (e.g. fracstrain) to distinguish 'good' and 'bad' designs 
        fracture for pristine 0.25632206
    nfeatures: int 
        number of features (length of 2D grid that flatten into 1D array)

    Return
    -----------
    x: matrix 
        matrix X (nsamples, nfeatures)

    y: 1D array
        y values (nsamples,)

    """
    y = np.zeros(len(data))

    count = 0
    for i in range(len(data)):
        if data[i][nfeatures+prop] > cutoff:
            y[i] = 1
            count += 1
        else:
            y[i] = 0

        if discrete == False:
            y[i] = data[i][nfeatures+prop]

    x = data[:, 0:nfeatures]

    print("Number of good designs "+str(count)+" out of total "+str(len(y)))
    return x, y

def split_data(x, y, frac_training, frac_test):
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
    frac_valid = 1 - frac_training - frac_test
    ntrain = int(frac_training * len(x))
    nvalid = int(frac_valid * len(x))
    # ntest = int(frac_test * len(x))

    X_train = x[:ntrain]
    X_valid = x[ntrain:ntrain+nvalid]
    X_test = x[ntrain+nvalid:]

    y_train = y[:ntrain]
    y_valid = y[ntrain:ntrain+nvalid]
    y_test =  y[ntrain+nvalid:]
    # [:,np.newaxis] to convert it to column array
    return X_train, X_valid, X_test, y_train[:,np.newaxis], y_valid[:,np.newaxis], y_test[:,np.newaxis]


raw_data = np.loadtxt('../raw/alldata_15G.dat', comments="#")
# [0-14]: Kirigami structure, '1': intact, '0': cut
# 15: Yield strain
# 16: Toughness (integration stress-strain curver up to yield point)
# 17: Yield stress

# paramters to make finer grids
n_corse_x = 3
n_corse_y = 5
n_fine_x = 30
n_fine_y = 80

listFG = []  # List Fine Grid
for i in range(len(raw_data)):
    kirigami_config = raw_data[i, 0:15]
    inner_wCuts = makeFineGrid(
        kirigami_config, n_corse_x, n_corse_y, n_fine_x, n_fine_y)
    listFG.append(inner_wCuts)

alldata_FG = np.array(listFG)
alldata_FG = np.append(alldata_FG, raw_data[:, -3:], 1)
# the last three columns are yield strain, toughness, and yield stress.
# just ignore the toughness

# the rest are TF, I will use the 15grid as an example here
#alldata = alldata_15G
alldata = alldata_FG  # unflag this for fine grid

nfeatures = len(alldata[0])-3  # nfeautures is needed later to split the matrix
print("Number of data:", len(alldata))
print("Number of features (or inputs/grids):", nfeatures)


# Create Train Test Split for Strain
x, y = create_matrix(alldata, False, 0, 0.375, nfeatures)
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(x, y, 0.5, 0.1)

# Create Train Test Split for Stress
x, y = create_matrix(alldata, False, 2, 0.375, nfeatures)
X_train, X_valid, X_test, y_train2, y_valid2, y_test2 = split_data(
    x, y, 0.5, 0.1)


print(len(y_valid), len(y_test))

# Visualize
# h = 30
# w = 80
for i in range(4):
    plt.imshow(X_train[np.random.randint(len(X_train))].reshape([n_fine_x, n_fine_y]).T)
    plt.show()

# calculate density
density = np.zeros((len(X_train), 1))
for i in range(len(X_train)):
    density[i] = (2400 - np.sum(X_train[i])/3/16)

# Plot of training dataset
plt.scatter(y_train, y_train2, c=density, cmap=plt.cm.Spectral, s=15)
plt.xlabel('Yield Strain')
plt.ylabel('Yield Stress')
plt.xlim([0.15, 1.5])
plt.xticks(np.arange(0.1, 2, step=0.2))
plt.ylim([5, 110])
plt.yticks(np.arange(10, 120, step=10))
plt.show()
