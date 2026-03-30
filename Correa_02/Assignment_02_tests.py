import numpy as np
# Modify the line below based on your last name

from Correa_02_01 import multi_layer_nn_torch

# These test modules have been converted from Tensorflow to PyTorch by  Islam, SM Mazharul. 2024_09_19
def get_data():
    x_train = np.array([
        [0.685938, -0.5756752], [0.944493, -0.02803439], [0.9477775, 0.59988844], [0.20710745, -0.12665261],
        [-0.08198895, 0.22326154], [-0.77471393, -0.73122877], [-0.18502127, 0.32624513],
        [-0.03133733, -0.17500992], [0.28585237, -0.01097354], [-0.19126464, 0.06222228],
        [-0.0303282, -0.16023481], [-0.34069192, -0.8288299], [-0.20600465, 0.09318836],
        [0.29411194, -0.93214977], [-0.7150941, 0.74259764], [0.13344735, 0.17136675],
        [0.31582892, 1.0810335], [-0.22873795, 0.98337173], [-0.88140666, 0.05909261],
        [-0.21215424, -0.05584779]
    ], dtype=np.float32)

    y_train = np.array([
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
        [0.0, 1.0], [1.0, 0.0]
    ], dtype=np.float32)

    return x_train, y_train


def get_data_2():
    x_train = np.array([
        [0.55824741, 0.8871946, 0.69239914], [0.25242493, 0.77856301, 0.66000716], [0.4443564, 0.1092453, 0.96508663],
        [0.66679551, 0.49591846, 0.9536062], [0.07967996, 0.61238854, 0.89165257],
        [0.36541977, 0.02095794, 0.49595849], [0.56918241, 0.45609922, 0.05487656],
        [0.38711358, 0.02771098, 0.27910454], [0.16556168, 0.9003711, 0.5345797], [0.70774465, 0.5294432, 0.77920751]
    ], dtype=np.float32)

    y_train = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    ], dtype=np.float32)

    return x_train, y_train


def test_random_weight_init():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0, loss_func='mse', seed=7321)
    assert weight_mat[0].dtype == np.float32
    assert weight_mat[1].dtype == np.float32
    assert weight_mat[0].shape == (3, 8)
    assert weight_mat[1].shape == (9, 2)
    assert np.allclose(weight_mat[0], np.array([
        [-0.5584747, -0.93098843, 0.97262794, 0.79936266, -0.04607159, -0.6551994, -0.02123384, 1.2985212],
        [-0.27985808, 0.6699221, 0.5720614, 0.02735023, 0.08965107, 1.3134009, 0.87998205, 0.64552426],
        [-0.7120164, -1.012345, -0.7378442, -0.1433691, 1.1782826, 0.91768885, 0.44304183, 0.05774327]
    ], dtype=np.float32))
    assert np.allclose(weight_mat[1], np.array([
        [-0.5584747, -0.93098843], [0.97262794, 0.79936266], [-0.04607159, -0.6551994], [-0.02123384, 1.2985212], [-0.27985808,  0.6699221],
        [0.5720614, 0.02735023], [0.08965107, 1.3134009], [0.87998205, 0.64552426], [-0.7120164, -1.012345]
    ], dtype=np.float32))


def test_weight_update_mse():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss_func='mse', seed=7321)
    assert np.allclose(weight_mat[0], np.array([
        [-0.5549652, -0.9315406, 0.97904986, 0.7971351, -0.04031776, -0.6551933, -0.0124495, 1.2785026],
        [-0.2806137, 0.66964424, 0.57033527, 0.0262097, 0.08960451, 1.3134062, 0.8827231, 0.6461926],
        [-0.7149695, -1.0119352, -0.73741806, -0.14267093, 1.1794946, 0.9176092, 0.44293016, 0.05870698]
    ], dtype=np.float32))
    assert np.allclose(weight_mat[1], np.array([
        [-5.3786999e-01, -9.2570597e-01], [9.7290343e-01, 7.9947048e-01], [-4.5702610e-02, -6.5509290e-01], [6.5118074e-04, 1.3026866e+00],
        [-2.6310697e-01, 6.7406589e-01], [5.7402265e-01, 2.8698910e-02], [9.0644889e-02, 1.3132796e+00], [8.8236296e-01, 6.4564961e-01], [-6.8476331e-01, -1.0063165e+00]
    ], dtype=np.float32))


def test_weight_update_ce():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss_func='CrossEntropy', seed=7321)
    assert np.allclose(weight_mat[0], np.array([
        [-0.55851483, -0.9310593, 0.96805966, 0.79607505, -0.04513008, -0.6550337, -0.0208312, 1.2995608],
        [-0.2798425, 0.66988623, 0.57183766, 0.02718921, 0.08959896, 1.3135577, 0.8800064, 0.64557517],
        [-0.7119839, -1.0122926, -0.7378986, -0.14340822, 1.1783141, 0.91774684, 0.44306424, 0.05775564]
    ], dtype=np.float32))
    assert np.allclose(weight_mat[1], np.array([
        [-0.5550133, -0.93444985], [0.9725987, 0.7993919], [-0.04608979, -0.6551812], [-0.01780057, 1.2950879],
        [-0.2770924, 0.66715646], [0.5720412, 0.02737045], [0.08952795, 1.313524], [0.88007927, 0.64542705], [-0.70740986, -1.0169516]
    ], dtype=np.float32))


def test_weight_update_svm():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss_func='svm', seed=7321)
    assert np.allclose(weight_mat[0], np.array([
        [-0.5577253, -0.9313979, 0.97500294, 0.7998317, -0.04516065, -0.654789, -0.01973041, 1.2946205],
        [-0.28006324, 0.6697214, 0.5721002, 0.02737862, 0.08955813, 1.3137885, 0.8804607, 0.64551663],
        [-0.712639, -1.0120362, -0.73880184, -0.14389704, 1.1784453, 0.91767734, 0.44299072, 0.058398]
    ], dtype=np.float32))
    assert np.allclose(weight_mat[1], np.array([
        [-0.5556622, -0.92911345], [0.97262794, 0.7994654], [-0.04607159, -0.6550991], [-0.01860914, 1.3009043], [-0.2776288, 0.67152715],
        [0.5722972, 0.02759097], [0.08965107, 1.3135757], [0.8801408, 0.64587575], [-0.7083771, -1.0099337]
    ], dtype=np.float32))


def test_assign_weights_by_value():
    np.random.seed(7321)
    x_train, y_train = get_data()
    init_w_0 = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    ], dtype=np.float32)
    init_w_1 = np.array([
        [0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0],
        [10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]
    ], dtype=np.float32)
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[init_w_0, init_w_1], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0, loss_func='CrossEntropy', seed=7321)
    assert np.allclose(weight_mat[0], init_w_0)
    assert np.allclose(weight_mat[1], init_w_1)


def test_error_output_dimensions():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=1, loss_func='mse', val_split=[0.5, 0.7], seed=7321)
    assert np.allclose(error, np.array([1.22728765])) or np.allclose(error, [1.22728765])
    np.random.seed(7321)
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=3, loss_func='mse', val_split=[0.5, 1.0], seed=7321)
    assert np.allclose(error, np.array([1.29949081, 1.23329556, 1.17263877])) or np.allclose(error, [1.29949081, 1.23329556, 1.17263877])


def test_error_vals_mse():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4, loss_func='mse', val_split=[0.5, 1.0], seed=7321)
    assert np.allclose(error, np.array([1.29949081, 1.23329556, 1.17263877, 1.11691725])) or np.allclose(error, [1.29949081, 1.23329556, 1.17263877, 1.11691725])
    np.random.seed(7321)
    x_train, y_train = get_data_2()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[7, 3], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=4, loss_func='mse', val_split=[0.5, 1.0], seed=7321)
    assert np.allclose(error, np.array([1.22968423, 1.13611138, 1.05420506, 0.98193675])) or np.allclose(error, [1.22968423, 1.13611138, 1.05420506, 0.98193675])


def test_error_vals_ce():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 9, 2], activations=['relu', 'relu', 'linear'], alpha=0.1, batch_size=32, epochs=4, loss_func='CrossEntropy', val_split=[0.5, 1.0], seed=7321)
    assert np.allclose(error, np.array([1.87873435, 1.70237231, 1.78377724, 1.73486936])) or np.allclose(error, [1.87873435, 1.70237231, 1.78377724, 1.73486936])
    np.random.seed(7321)
    x_train, y_train = get_data_2()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[7, 9, 3], activations=['relu', 'relu', 'linear'], alpha=0.1, batch_size=32, epochs=4, loss_func='CrossEntropy', val_split=[0.5, 1.0], seed=7321)
    assert np.allclose(error, np.array([2.09394574, 1.68945384, 1.79011643, 1.35292614])) or np.allclose(error, [2.09394574, 1.68945384, 1.79011643, 1.35292614])


def test_validation_output():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=0, loss_func='mse', val_split=[0.5, 1.0], seed=7321)
    assert len(error) == 0

    np.random.seed(7321)
    weight_mat, error, output = multi_layer_nn_torch(x_train=x_train, y_train=y_train, layers=[8, 2], activations=['relu', 'linear'], alpha=0.01, batch_size=32, epochs=5, loss_func='mse', val_split=[0.5, 1.0], seed=7321)
    assert output.shape == (10, 2)
    assert np.allclose(output, np.array([
        [-1.2492647, -0.11274791], [-0.9644941, 0.71884865], [-1.1652137, -0.41964224], [-1.2865348, 0.6934375],
        [-0.6155797, -1.1980405], [-1.0144439, -0.31920582], [0.07424253, 0.04851192], [-0.3764736, -1.2368815], [-0.9748718, -0.46610907],
        [-1.1943675, -0.25345165]
    ], dtype=np.float32))


def test_many_layers():
    np.random.seed(7321)
    x_train, y_train = get_data()
    weight_mat, error, output = multi_layer_nn_torch(
        x_train=x_train, y_train=y_train, layers=[8, 6, 7, 5, 3, 1, 9, 2],
        activations=['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        alpha=0.01, batch_size=32, epochs=2, loss_func='CrossEntropy', seed=7321)
    assert weight_mat[0].shape == (3, 8)
    assert weight_mat[1].shape == (9, 6)
    assert weight_mat[2].shape == (7, 7)
    assert weight_mat[3].shape == (8, 5)
    assert weight_mat[4].shape == (6, 3)
    assert weight_mat[5].shape == (4, 1)
    assert weight_mat[6].shape == (2, 9)
    assert weight_mat[7].shape == (10, 2)
    assert output.shape == (4, 2)
    assert (isinstance(error, np.ndarray) or isinstance(error, list)) and len(error) == 2
