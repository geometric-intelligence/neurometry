# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Grid score calculations."""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """Calculating the grid scores with different radius."""
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: (b and in_val) or out_val)
    return vfunc(z)


class GridScorer:
    """Class for scoring ratemaps given trajectories."""

    def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
        """Scoring ratemaps given trajectories.
        Args:
          nbins: Number of bins per dimension in the ratemap.
          coords_range: Environment coordinates range.
          mask_parameters: parameters for the masks that analyze the angular
            autocorrelation of the 2D autocorrelation.
          min_max: Correction.
        """
        self._nbins = nbins
        self._min_max = min_max
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        # Create all masks
        self._masks = [
            (self._get_ring_mask(mask_min, mask_max), (mask_min, mask_max))
            for mask_min, mask_max in mask_parameters
        ]
        # Mask for hiding the parts of the SAC that are never used
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan,
        )

    def calculate_ratemap(self, xs, ys, activations, statistic="mean"):
        return scipy.stats.binned_statistic_2d(
            xs,
            ys,
            activations,
            bins=self._nbins,
            statistic=statistic,
            range=self._coords_range,
        )[0]

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return circle_mask(n_points, mask_max * self._nbins) * (
            1 - circle_mask(n_points, mask_min * self._nbins)
        )

    def grid_score_60(self, corr):
        if self._min_max:
            return np.minimum(corr[60], corr[120]) - np.maximum(
                corr[30], np.maximum(corr[90], corr[150])
            )
        return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    def grid_score_90(self, corr):
        return corr[90] - (corr[45] + corr[135]) / 2

    def calculate_sac(self, seq1):
        """Calculating spatial autocorrelogram."""
        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode="full")

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        std_seq1 = np.power(
            np.subtract(
                np.divide(sum_seq1_sq, n_bins),
                (np.divide(np.square(sum_seq1), n_bins_sq)),
            ),
            0.5,
        )
        std_seq2 = np.power(
            np.subtract(
                np.divide(sum_seq2_sq, n_bins),
                (np.divide(np.square(sum_seq2), n_bins_sq)),
            ),
            0.5,
        )
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins),
            np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq),
        )
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        return np.nan_to_num(x_coef)

    def rotated_sacs(self, sac, angles):
        return [
            scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
            for angle in angles
        ]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        # Calculate dc on the ring area
        masked_sac_mean = np.sum(masked_sac) / ring_area
        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs, strict=False):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

    def get_scores(self, rate_map):
        """Get summary of scrores for grid cells."""
        sac = self.calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        scores = [
            self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for mask, mask_params in self._masks  # pylint: disable=unused-variable
        ]
        scores_60, scores_90, variances = map(
            np.asarray, zip(*scores, strict=False)
        )  # pylint: disable=unused-variable
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)

        return (
            scores_60[max_60_ind],
            scores_90[max_90_ind],
            self._masks[max_60_ind][1],
            self._masks[max_90_ind][1],
            sac,
            max_60_ind,
        )

    def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):
        """Plot ratemaps."""
        if ax is None:
            ax = plt.gca()
        ax.imshow(ratemap, *args, interpolation="none", **kwargs)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)

    def plot_sac(self, sac, mask_params=None, ax=None, title=None, *args, **kwargs):
        """Plot spatial autocorrelogram."""
        if ax is None:
            ax = plt.gca()
        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, *args, interpolation="none", **kwargs)
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[0] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor="k",
                )
            )
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[1] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor="k",
                )
            )
        ax.axis("off")
        if title is not None:
            ax.set_title(title)

    def border_score(self, rm, res, box_width):
        # Find connected firing fields
        pix_area = 100**2 * box_width**2 / res**2
        rm_thresh = rm > (rm.max() * 0.3)
        rm_comps, ncomps = ndimage.measurements.label(rm_thresh)

        # Keep fields with area > 200cm^2
        masks = []
        nfields = 0
        for i in range(1, ncomps + 1):
            mask = (rm_comps == i).reshape(res, res)
            if mask.sum() * pix_area > 200:
                masks.append(mask)
                nfields += 1

        # Max coverage of any one field over any one border
        cm_max = 0
        for mask in masks:
            mask = masks[0]
            n_cov = mask[0].mean()
            s_cov = mask[-1].mean()
            e_cov = mask[:, 0].mean()
            w_cov = mask[:, -1].mean()
            cm = np.max([n_cov, s_cov, e_cov, w_cov])
            if cm > cm_max:
                cm_max = cm

        # Distance to nearest wall
        x, y = np.mgrid[:res, :res] + 1
        x = x.ravel()
        y = y.ravel()
        xmin = np.min(np.vstack([x, res + 1 - x]), 0)
        ymin = np.min(np.vstack([y, res + 1 - y]), 0)
        dweight = np.min(np.vstack([xmin, ymin]), 0).reshape(res, res)
        dweight = dweight * box_width / res

        # Mean firing distance
        dms = []
        for mask in masks:
            field = rm[mask]
            field /= field.sum()  # normalize
            dm = (field * dweight[mask]).sum()
            dms.append(dm)
        dm = np.nanmean(dms) / (box_width / 2)
        border_score = (cm_max - dm) / (cm_max + dm)
        return border_score, cm_max, dm

    def band_score(self, rm, res, box_width):
        """Get band score"""
        X = np.linspace(0.0, box_width, res)
        Y = np.linspace(0.0, box_width, res)
        k = np.arange(0.0, 2, 0.1)
        r2 = []

        for ii in range(np.shape(k)[0]):
            for jj in range(np.shape(k)[0]):
                Z = np.outer(
                    np.exp(1j * 2 * np.pi * k[ii] * X),
                    np.exp(1j * 2 * np.pi * k[jj] * Y),
                )
                r2.append(np.corrcoef(np.real(Z).flatten(), rm.flatten())[0, 1])

        return np.nanmax(r2)

    def get_sac_interp(self, cell):
        """Get interpolated sac."""
        sac = self.calculate_sac(cell)
        xx = np.linspace(-1, 1, 99)
        yy = np.linspace(-1, 1, 99)
        return scipy.interpolate.RegularGridInterpolator((xx, yy), sac)

    def get_phi(self, cell, interp=None, spacing_values=None):  # 0.15
        """Get orientation of grid cell."""

        if spacing_values is None:
            spacing_values = np.arange(0.01, 1.0, 0.01)
        if interp is None:
            interp = self.get_sac_interp(cell)

        n_angles = 1000
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        sum_vec = []
        radial_values = []
        for r in spacing_values:
            values = interp(np.array([r * np.sin(angles), r * np.cos(angles)]).T)
            sum_vec.append(np.sum(values))
            radial_values.append(values)

        # radial_values = np.mean(radial_values, axis=0)
        peaks_sum, _ = scipy.signal.find_peaks(sum_vec)
        radial_values = radial_values[peaks_sum[0]]
        peaks_grids, _ = scipy.signal.find_peaks(radial_values, distance=n_angles / 8)
        if peaks_grids.size >= 5:
            phi = angles[peaks_grids][:3]
        else:
            phi = np.zeros((3,))
            # print no phi found and the cell number
            print(f"no 6 angles found for cell {cell}")

        return phi, radial_values

    def get_spacing(self, cell, interp=None, phi=None):
        """Get spacing of grid cell. If no phi is given, it will take the first phi"""

        # if both interp and phi are not given, calculate them
        if interp is None:
            interp = self.get_sac_interp(cell)

        if phi is None:
            phi, _ = self.get_phi(cell, interp)
            phi = phi[0]

        if phi < (np.pi / 4):
            scaling = 1 / np.cos(phi)
        elif phi < (np.pi / 2):
            scaling = 1 / np.cos((np.pi / 2) - phi)
        elif phi < (3 * np.pi / 4):
            scaling = 1 / np.cos(phi - (np.pi / 2))
        elif phi < np.pi:
            scaling = 1 / np.cos(np.pi - phi)
        elif phi < (5 * np.pi / 4):
            scaling = 1 / np.cos(phi - np.pi)

        spacing_vec = np.linspace(0.001, scaling - 0.01, 1000)
        spacing_values = []
        for r in spacing_vec:
            value = interp(np.array([r * np.sin(phi), r * np.cos(phi)]).T)
            spacing_values.append(value)

        spacing_values = np.array(spacing_values)
        spacing_peaks, _ = scipy.signal.find_peaks(
            spacing_values[:, 0], prominence=0.05
        )

        return 0 if spacing_peaks.size == 0 else 2 * spacing_vec[spacing_peaks][0]
