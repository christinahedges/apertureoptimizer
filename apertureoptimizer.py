'''Optimizes apertures for Kepler, K2 and TESS'''

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, Box1DKernel
from tqdm import tqdm
from copy import deepcopy
import warnings

class ApertureOptimizerError(Exception):
    '''Raised if there is a problem...'''
    pass

class ApertureOptimizer(object):
    '''Class to optimze apertures
    '''
    def _transit_mask(self, time, period, t0, duration):
        '''Builds a transit mask. Lifted from bls.py'''
        hp = 0.5 * period
        mask = ~(np.abs((time - t0 + hp) % period - hp) < 0.5 * duration)
        return mask

    def __init__(self, tpf, period, t0, duration, corrector=None):
        self.tpf = tpf
        self.Y, self.X = np.meshgrid(np.arange(0, tpf.shape[1]), np.arange(0, tpf.shape[2]))
        # Mask where TPF actually has data.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.tpf_has_data = np.isfinite(np.nanmedian(tpf.flux, axis=0))
        self.period = period
        self.t0 = t0
        self.duration = duration
        if hasattr(corrector, '__call__'):
            self.corrector = corrector
        else:
            self.corrector = lambda x: x
        self.mask = lambda x: self._transit_mask(x, self.period, self.t0, self.duration)
        if not (~self.mask(self.tpf.time)).any():
            raise ApertureOptimizerError('No transits found.')
        self.starting_mask = self._find_starting_pixel()
        self.starting_lc = self.corrector(self.tpf.to_lightcurve())


    def __repr__(self):
        return ('ApertureBuilder Class. TPF ID {}, Period:{}, t0:{}, duration:{}'
                ''.format(self.tpf.targetid, self.period, self.t0, self.duration))


    def _measure_snr(self, clc):
        ''' Measure the signal to noise for a given light curve
        '''
        mask = ~self.mask(clc.time)

        # Find errors...
        intransit_error = (np.nansum(clc.flux_err[mask]**2)**0.5)/float(np.sum(mask))
        if np.sum(~mask) != 0:
            outtransit_error = (np.nansum(clc.flux_err[~mask]**2)**0.5)/float(np.sum(~mask))
        else:
            outtransit_error = 0
        average_error = (intransit_error**2 + outtransit_error**2)**0.5

        # Measure depth
        depth = np.nanmean(clc.flux[~mask]) - np.nanmean(clc.flux[mask])


#        error = (average_error**2 + np.std(clc.flux[~mask] - depth)**2)**0.5
        error = average_error
        snr = depth/error
        return snr



    def _find_starting_pixel(self):
        '''Measures the signal to noise in a given time mask for all pixels in a tpf

        Returns
        -------
        aper : np.ndarray with shape of tpf pixels
            Boolean mask with the best starting pixel
        '''
        snr = np.zeros(len(np.where(self.tpf_has_data)[0]), dtype=float)
        for kdx, idx, jdx in zip(range(self.tpf.shape[1] * self.tpf.shape[2]), np.where(self.tpf_has_data)[0], np.where(self.tpf_has_data)[1]):
            aper = np.zeros(self.tpf.shape[1:], dtype=bool)
            aper[idx, jdx] = True
            lc = self.tpf.to_lightcurve(aperture_mask=aper).normalize()
            clc = self.corrector(lc)
            snr[kdx] = self._measure_snr(clc)

        best = np.argmax(snr)
        idx, jdx = np.where(self.tpf_has_data)[0][best], np.where(self.tpf_has_data)[1][best]
        aper = np.zeros(self.tpf.shape[1:], dtype=bool)
        aper[idx, jdx] = True
        return aper

    def _find_neighbours(self, aper):
        '''Finds the neighbouring pixels, given a 2D boolean maskself.

        Note: Diagonal pixels are considered adjacent.

        Parameters
        ----------
        aper : np.ndarray with shape of tpf pixels
            Boolean mask, where True is a pixel used in a mask

        Returns
        -------
        pixels : np.ndarray with size 2 x number of neighbours
            All the neighbouring pixels to the aper mask
        '''
        # Find the neighbours
        ndx = np.empty(0, dtype=int)
        for m in np.asarray(np.where(aper)).T[:, [1,0]]:
            dist = ((np.abs(m[0] - self.X)**2 + np.abs(m[1] - self.Y)**2)**0.5)
            ndx = np.append(ndx, np.asarray(np.where(dist < 2), dtype=int).T)
        ndx = ndx.reshape((len(ndx)//2, 2))
        ndx = np.unique(ndx, axis=0)

        # Mask out some bad values
        mask = np.zeros(len(ndx), dtype=bool)
        # No duplicates
        for m in np.asarray(np.where(aper)).T[:, [1, 0]]:
            mask |= (ndx == m).all(axis=1)
        # No nan values
        for m in np.asarray(np.where(~self.tpf_has_data)).T[:, [1, 0]]:
            mask |= (ndx == m).all(axis=1)
        ndx = ndx[~mask].T
        pixels = ndx[[1, 0], :]
        return pixels


    def _get_neighbour_boolean_mask(self, aper):
        '''Find neighbours and return as a boolean mask.

        Just for visuals really.

        Parameters
        ----------
        aper : np.ndarray with shape of tpf pixels
            Boolean mask, where True is a pixel used in a mask

        Returns
        -------
        new_aper : np.ndarray with shape of tpf pixels
            Boolean mask, where True is a neighbour pixel.
        '''
        neighbours = self._find_neighbours(aper)
        new_aper = np.zeros(self.tpf.shape[1:], dtype=bool)
        for x, y in neighbours.T:
            new_aper[x, y] = True
        return new_aper


    def optimize(self, n=None):
        '''Find the optimal aperture

        Parameters
        ----------
        n : int
            Number of pixels to timeout at. If None, will try all pixels.

        '''
        used = deepcopy(self.starting_mask)
        if n is None:
            n = self.tpf.shape[1] * self.tpf.shape[2]
        best_snr = np.zeros(n)
        apers = np.zeros((n, self.tpf.shape[1], self.tpf.shape[2]), dtype=bool)
        for count in tqdm(np.arange(0, n), desc='Searching Neighbours...'):
            pixels = self._find_neighbours(used).T
            if len(pixels) == 0:
                break
            snr = np.zeros(len(pixels))
            for idx, n in enumerate(pixels):
                aper = np.copy(used)
                aper[n[0], n[1]] = True
                lc = self.tpf.to_lightcurve(aperture_mask=aper).normalize()
                clc = self.corrector(lc)
                snr[idx] = self._measure_snr(clc)


            best = pixels[snr.argmax()]
            used[(best[0], best[1])] = True
            best_snr[count] = np.max(snr)
            apers[count] = np.asarray(used, dtype=bool)

        best = best_snr.argmax()
        self.apers = apers
        self.snr = best_snr
        self.best_idx = best
        self.best_aper = apers[best]
        self.best_lc = self.corrector(self.tpf.to_lightcurve(aperture_mask=self.best_aper).normalize())
#        return apers[best]


    def plot_results(self, fig=None, bin=20):
        '''Plot up the results.
        '''
        if not hasattr(self, 'best_lc'):
            raise AttributeError('Please run optimze before plotting results.')
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot2grid((2,2), (0,0), fig=fig)
        self.tpf.plot(aperture_mask=self.tpf.pipeline_mask, ax=ax, show_colorbar=False)
        ax.set_title("Pipeline Aperture")
        ax = plt.subplot2grid((2,2), (0,1))
        self.tpf.plot(aperture_mask=self.best_aper, ax=ax, show_colorbar=False)
        ax.set_title("Optimal Aperture")
        ax.set_ylabel('')
        ax = plt.subplot2grid((2,2), (1,0), colspan=2)
        pipeline_lc = self.corrector(self.tpf.to_lightcurve()).fold(self.period, self.t0).bin(bin)
        pipeline_lc.errorbar(label='', c='k', ax=ax)
        pipeline_lc.plot(ax=ax, label='Pipeline Aperture', c='k')
        best = self.corrector(self.tpf.to_lightcurve(aperture_mask=self.best_aper)).fold(self.period, self.t0).bin(bin)
        best.errorbar(ax=ax, label='', c='r')
        best.plot(ax=ax, label='Optimal Aperture', c='r')
        return fig
