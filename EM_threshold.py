from typing import Literal, Optional, Tuple, Union
import numpy as np
from sklearn.mixture import GaussianMixture
import dask
import dask.array as da
from skimage.filters import threshold_otsu

def select_water_tiles_buffer(tiles: Union[np.ndarray, np.ma.MaskedArray],
                      hand_threshold: float, hand_fraction: list) -> np.ndarray:
    if np.allclose(tiles, 0.0):
        return []
    tile_indexes = np.arange(tiles.shape[0])
    if np.count_nonzero(tiles>=hand_threshold) == 0:
        return []
    else:
        percent_valid_pixels = np.sum(tiles, axis=(1, 2)) / (tiles.shape[1] * tiles.shape[2])
        return tile_indexes[(percent_valid_pixels >= hand_fraction[0]) & (percent_valid_pixels <= hand_fraction[1])]

def select_water_tiles(tiles: Union[np.ndarray, np.ma.MaskedArray],
                      hand_threshold: float, hand_fraction: list) -> np.ndarray:
    if np.allclose(tiles, 0.0):
        raise ValueError(f'All pixels in scene have a Persistent Water occurrence of {0.0} (all water); '
                         f'scene is not a good candidate for water mapping.')

    tile_indexes = np.arange(tiles.shape[0])

    tiles = np.ma.masked_greater_equal(tiles, hand_threshold)
    if np.count_nonzero(tiles>=hand_threshold) == 0:
        return []
    else:
        percent_valid_pixels = np.sum(tiles.mask, axis=(1, 2)) / (tiles.shape[1] * tiles.shape[2])
        return tile_indexes[(percent_valid_pixels >= hand_fraction[0]) & (percent_valid_pixels <= hand_fraction[1])]

def mean_of_subtiles(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 8, tiles.shape[2] // 8)
    sub_tiles_mean = np.zeros((tiles.shape[0], 64))
    for ii, tile in enumerate(tiles):
        sub_tiles = tile_array(tile.filled(0), tile_shape=sub_tile_shape)
        sub_tiles_mean[ii, :] = sub_tiles.mean(axis=(1, 2))
    return sub_tiles_mean

def mean_of_subtiles_dask(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 8, tiles.shape[2] // 8)
    tiles_dask = da.from_array(tiles, chunks=(1, None, None))
    sub_tiles_mean = tiles_dask.map_blocks(
        lambda tile: tile_array(tile.filled(0), tile_shape=sub_tile_shape).mean(axis=(1, 2)),
        dtype=tiles.dtype,
        chunks=(1, 64)
    ).compute()
    return sub_tiles_mean

def select_backscatter_tiles(backscatter_tiles: np.ndarray, hand_candidates: np.ndarray, lower_median = False) -> np.ndarray:
    '''
    Tile selection for the back up threshold.
    '''
    tile_indexes = np.arange(backscatter_tiles.shape[0])
    sub_tile_means = mean_of_subtiles(backscatter_tiles)
    sub_tile_means_std = sub_tile_means.std(axis=1)
    tile_medians = np.ma.median(backscatter_tiles, axis=(1, 2))
    tile_variance = sub_tile_means_std / tile_medians

    low_mean_threshold = np.ma.median(tile_medians[hand_candidates])
    low_mean_candidates = tile_indexes[tile_medians < low_mean_threshold]
    if lower_median:
        potential_candidates = np.intersect1d(hand_candidates, low_mean_candidates)
    else:
        potential_candidates = np.intersect1d(hand_candidates, tile_indexes)
    for variance_threshold in np.nanpercentile(tile_variance.filled(np.nan), np.arange(1, 96)[::-1]):
        variance_candidates = tile_indexes[tile_variance > variance_threshold]
        selected = np.intersect1d(variance_candidates, potential_candidates)
        sort_index = np.argsort(sub_tile_means_std[selected])[::-1]
        if len(selected) >= 10:
            #print(f'Found 10 tiles with variance > {variance_threshold} and median backscatter < {low_mean_threshold}')
            return selected[sort_index][:10]
    return np.array([])

def tile_array(array: Union[np.ndarray, np.ma.MaskedArray], tile_shape: Tuple[int, int] = (200, 200),
               pad_value: float = None) -> Union[np.ndarray, np.ma.MaskedArray]:
    '''
    Split an array into tiles of a given shape.
    Non overlapping version
    '''
    array_rows, array_columns = array.shape
    tile_rows, tile_columns = tile_shape

    rpad = -array_rows % tile_rows
    cpad = -array_columns % tile_columns

    if (rpad or cpad) and pad_value is None:
        raise ValueError(f'Cannot evenly tile a {array.shape} array into ({tile_rows},{tile_columns}) tiles')

    if rpad or cpad:
        padded_array = np.pad(array, ((0, rpad), (0, cpad)), constant_values=pad_value)
        if isinstance(array, np.ma.MaskedArray):
            mask = np.pad(array.mask, ((0, rpad), (0, cpad)), constant_values=True)
            padded_array = np.ma.MaskedArray(padded_array, mask=mask)
    else:
        padded_array = array

    tile_list = []
    for rows in np.vsplit(padded_array, range(tile_rows, array_rows, tile_rows)):
        tile_list.extend(np.hsplit(rows, range(tile_columns, array_columns, tile_columns)))

    dstack = np.ma.dstack if isinstance(array, np.ma.MaskedArray) else np.dstack
    tiled = np.moveaxis(dstack(tile_list), -1, 0)

    return tiled

def determine_em_threshold_Otsu(tiles: np.ndarray, scaling: float) -> float:
    thresholds = []
    for ii in range(tiles.shape[0]):
        test_tile = (np.around(tiles[ii, :, :] * scaling)).astype(int)
        thresholds.append(thre_otsu(test_tile) / scaling)
        #np.median(np.sort(thresholds)) np.max(np.sort(thresholds)) np.percentile(thresholds,25)
    return np.median(thresholds)

def thre_otsu(tile):
    threshold = threshold_otsu(tile)
    return threshold

def determine_em_threshold(tiles: np.ndarray, scaling: float) -> float:
    thresholds = []
    for ii in range(tiles.shape[0]):
        test_tile = (np.around(tiles[ii, :, :] * scaling)).astype(int)
        thresholds.append(expectation_maximization_threshold(test_tile) / scaling)
        #np.median(np.sort(thresholds)) np.max(np.sort(thresholds)) np.percentile(thresholds,25)
    return np.median(thresholds)

def _make_histogram(image):
    image = image.flatten()
    indices = np.nonzero(np.isnan(image))
    image[indices] = 0
    indices = np.nonzero(np.isinf(image))
    image[indices] = 0
    del indices
    size = image.size
    maximum = int(np.ceil(np.amax(image)) + 1)
    histogram = np.zeros((1, maximum))
    for i in range(0, size):
        floor_value = np.floor(image[i]).astype(np.uint8)
        if 0 < floor_value < maximum - 1:
            temp1 = image[i] - floor_value
            temp2 = 1 - temp1
            histogram[0, floor_value] = histogram[0, floor_value] + temp1
            histogram[0, floor_value - 1] = histogram[0, floor_value - 1] + temp2
    histogram = np.convolve(histogram[0], [1, 2, 3, 2, 1])
    histogram = histogram[2:(histogram.size - 3)]
    histogram /= np.sum(histogram)
    return histogram

def _make_distribution(m, v, g, x):
    x = x.flatten()
    m = m.flatten()
    v = v.flatten()
    g = g.flatten()
    y = np.zeros((len(x), m.shape[0]))
    for i in range(0, m.shape[0]):
        d = x - m[i]
        amp = g[i] / np.sqrt(2 * np.pi * v[i])
        y[:, i] = amp * np.exp(-0.5 * (d * d) / v[i])
    return y

def expectation_maximization_threshold(tile: np.ndarray, number_of_classes: int = 2) -> float:
    """Water threshold Calculation using a multi-mode Expectation Maximization Approach

    Thresholding works best when backscatter tiles are provided on a decibel scale
    to get Gaussian distribution that is scaled to a range of 0-255, and performed
    on a small tile that is likely to have a transition between liquid water and others.

    Args:
        tile: array of backscatter values for a tile from an RTC raster
        number_of_classes: classify the tile into this many classes.
    Returns:
        threshold: threshold value in decibels
    """
    image_copy = tile.copy()
    image_copy2 = np.ma.filled(tile.astype(float), np.nan)  # needed for valid posterior_lookup keys
    image = tile.flatten()
    minimum = np.amin(image)
    image = image - minimum + 1
    maximum = np.amax(image)

    histogram = _make_histogram(image)
    nonzero_indices = np.nonzero(histogram)[0]
    histogram = histogram[nonzero_indices]
    histogram = histogram.flatten()
    class_means = (
            (np.arange(number_of_classes) + 1) * maximum /
            (number_of_classes + 1)
    )
    class_variances = np.ones(number_of_classes) * maximum
    class_proportions = np.ones(number_of_classes) * 1 / number_of_classes
    sml = np.mean(np.diff(nonzero_indices)) / 1000
    iteration = 0
    while True:
        class_likelihood = _make_distribution(
            class_means, class_variances, class_proportions, nonzero_indices
        )
        sum_likelihood = np.sum(class_likelihood, 1) + np.finfo(
            class_likelihood[0][0]).eps
        log_likelihood = np.sum(histogram * np.log(sum_likelihood))
        for j in range(0, number_of_classes):
            class_posterior_probability = (
                    histogram * class_likelihood[:, j] / sum_likelihood
            )
            class_proportions[j] = np.sum(class_posterior_probability)
            class_means[j] = (
                    np.sum(nonzero_indices * class_posterior_probability)
                    / class_proportions[j]
            )
            vr = (nonzero_indices - class_means[j])
            class_variances[j] = (
                    np.sum(vr * vr * class_posterior_probability)
                    / class_proportions[j] + sml
            )
            del class_posterior_probability, vr
        class_proportions += 1e-3
        class_proportions /= np.sum(class_proportions)
        class_likelihood = _make_distribution(
            class_means, class_variances, class_proportions, nonzero_indices
        )
        sum_likelihood = np.sum(class_likelihood, 1) + np.finfo(
            class_likelihood[0, 0]).eps
        del class_likelihood
        new_log_likelihood = np.sum(histogram * np.log(sum_likelihood))
        del sum_likelihood
        if (new_log_likelihood - log_likelihood) < 0.000001:
            break
        iteration += 1
    del log_likelihood, new_log_likelihood
    class_means = class_means + minimum - 1
    s = image_copy.shape
    posterior = np.zeros((s[0], s[1], number_of_classes))
    posterior_lookup = dict()
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            pixel_val = image_copy2[i, j]
            if pixel_val in posterior_lookup:
                for n in range(0, number_of_classes):
                    posterior[i, j, n] = posterior_lookup[pixel_val][n]
            else:
                posterior_lookup.update({pixel_val: [0] * number_of_classes})
                for n in range(0, number_of_classes):
                    x = _make_distribution(
                        class_means[n], class_variances[n], class_proportions[n],
                        image_copy[i, j]
                    )
                    posterior[i, j, n] = x * class_proportions[n]
                    posterior_lookup[pixel_val][n] = posterior[i, j, n]

    sorti = np.argsort(class_means)
    xvec = np.arange(class_means[sorti[0]], class_means[sorti[1]], step=.05)
    x1 = _make_distribution(class_means[sorti[0]], class_variances[sorti[0]], class_proportions[sorti[0]], xvec)
    x2 = _make_distribution(class_means[sorti[1]], class_variances[sorti[1]], class_proportions[sorti[1]], xvec)
    dx = np.abs(x1 - x2)

    return xvec[np.argmin(dx)]

def tile_array_overlap(array: np.ndarray, tile_shape: Tuple[int, int] = (80, 80), overlap: float = 0.2, pad_value: float = None) -> np.ndarray:
    """Tile a 2D numpy array with overlapping

    Args:
        array: 2D array to tile
        tile_shape: the shape of each tile
        overlap: overlap between tiles, expressed as a fraction of tile size

    Returns:
        the tiled array
    """
    array_rows, array_columns = array.shape
    tile_rows, tile_columns = tile_shape

    # Calculate the number of tiles along each axis
    num_tiles_x = int(np.ceil(array_rows / (tile_rows - overlap * tile_rows)))
    num_tiles_y = int(np.ceil(array_columns / (tile_columns - overlap * tile_columns)))

    # Calculate the padding needed to evenly tile the array
    rpad = num_tiles_x * tile_rows - array_rows
    cpad = num_tiles_y * tile_columns - array_columns
    
    if (rpad or cpad) and pad_value is None:
        raise ValueError(f'Cannot evenly tile a {array.shape} array into ({tile_rows},{tile_columns}) tiles')

    # Pad the array
    padded_array = np.pad(array, ((0, rpad), (0, cpad)), constant_values=0)

    # Initialize an empty array to store the tiles
    tiles = np.zeros((num_tiles_x * num_tiles_y, tile_rows, tile_columns))

    # Extract tiles
    tile_index = 0
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_start = int(i * (tile_rows - overlap * tile_rows))
            y_start = int(j * (tile_columns - overlap * tile_columns))
            x_end = x_start + tile_rows
            y_end = y_start + tile_columns
            tiles[tile_index] = padded_array[x_start:x_end, y_start:y_end]
            tile_index += 1

    return tiles

def select_tiles(backscatter_tiles: np.ndarray, hand_candidates: np.ndarray, lower_median = False) -> np.ndarray:
    '''
    Tile selection for the adaptive threshold that most likely the upper boundary of liquid water.
    '''
    # we should probably do this in log space
    tile_indexes = np.arange(backscatter_tiles.shape[0])
    sub_tile_means = mean_of_subtiles(backscatter_tiles)
    sub_tile_means_std = sub_tile_means.std(axis=1)
    tile_medians = np.ma.median(backscatter_tiles, axis=(1, 2))
    tile_variance = sub_tile_means_std / tile_medians
    tile_variance_water = tile_variance[hand_candidates]
    
    if lower_median:
        low_mean_threshold = np.ma.median(tile_medians[hand_candidates])
        low_mean_candidates = tile_indexes[tile_medians < low_mean_threshold]
        potential_candidates = np.intersect1d(hand_candidates, low_mean_candidates)
    else:
        potential_candidates = np.intersect1d(hand_candidates, tile_indexes)
    for variance_threshold in np.nanpercentile(tile_variance_water.filled(np.nan), np.arange(5, 25)[::-1]):
        variance_candidates = tile_indexes[tile_variance > variance_threshold]
        selected = np.intersect1d(variance_candidates, potential_candidates)
        if len(selected)>=5:
            selec_tiles = backscatter_tiles[selected,:,:]
            bimodal_index = bimodal_tiles(selec_tiles)
            true_indices = np.where(bimodal_index)[0]
            if len(true_indices) >= 5:
                return selected[true_indices]
    return np.array([])

def bimodal_tiles(tiles):
    '''
    select the tiles that show bimodal distribution
    '''
    pass_index = []
    for ii in range(tiles.shape[0]):
        #pass_index.append(bimodal_test(tiles[ii, :, :]))
        pass_index.append(bimodal_test(np.log10(tiles[ii, :, :])+ 30.))
    return dask.compute(*pass_index)
        
@dask.delayed
def bimodal_test_ori(arr):
    '''
    Test if a 2D array is bimodal distribution using a Gaussian Mixture Model and the weights of the two components
    '''
    gmm = GaussianMixture(n_components=2)
    gmm.fit(arr.reshape(-1, 1))
    weights = gmm.weights_
    if np.all(weights > 0.3) and np.all(weights < 0.7):
        return True
    else:
        return False

@dask.delayed
def bimodal_test(arr):
    '''
    Test if a 2D array is bimodal distribution using a Gaussian Mixture Model,
    the weights of the two components, and the Ashman D (AD) coefficient threshold of >2 for bimodality.
    '''
    gmm = GaussianMixture(n_components=2)
    gmm.fit(arr.reshape(-1, 1))
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    ad_coefficient = compute_ad(means, stds)
    
    # Perform Otsu thresholding on the array
    threshold = threshold_otsu(arr)
    
    # Check if the histogram value at the threshold is less than 0.2 of the peak value
    hist_peak = np.max(np.histogram(arr, bins='auto')[0])
    hist_value_at_threshold = np.histogram(arr, bins='auto')[0][int(np.digitize(threshold, np.histogram(arr, bins='auto')[1]))]
    
    if np.all(weights > 0.3) and np.all(weights < 0.7) and ad_coefficient > 1.5 and hist_value_at_threshold <= 0.3 * hist_peak:
        return True
    else:
        return False

def compute_ad(means, stds):
    '''
    Compute the Ashman D (AD) coefficient using the means and standard deviations of the Gaussian components.
    '''
    delta_mean = means[0] - means[1]
    std_sum = stds[0]**2 + stds[1]**2
    ad_coefficient = delta_mean / np.sqrt(0.5 * std_sum)
    return ad_coefficient