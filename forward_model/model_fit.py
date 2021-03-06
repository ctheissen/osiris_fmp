import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt
from astropy.io import fits
import osiris_fmp as nsp
import emcee
import corner
import copy
import time
import os
import sys
import scipy as sp


def makeModel(teff, logg, z, vsini, rv, alpha, wave_offset, flux_offset, flux_multiplier, **kwargs):
	"""
	Return a forward model.

	Parameters
	----------
	params : a dictionary that specifies the parameters such as teff, logg, z.
	data   : an input science data used for continuum correction

	Returns
	-------
	model: a synthesized model
	"""

	# read in the parameters
	modelset   = kwargs.get('modelset', 'aces2013')
	lsf        = kwargs.get('lsf', None)   # instrumental LSF
	pgs        = kwargs.get('pgs', None)  # pgs
	vsini_set  = kwargs.get('vsini_set', True) # apply telluric
	tell       = kwargs.get('tell', True) # apply telluric
	data       = kwargs.get('data', None) # for continuum correction and resampling
	instrument = kwargs.get('instrument', 'OSIRIS') # for continuum correction and resampling
	band       = kwargs.get('band', 'Kbb') # for continuum correction and resampling
	smooth     = kwargs.get('smooth', False) # for continuum correction and resampling
	JHK        = kwargs.get('JHK', False) # for continuum correction and resampling

	#print('Params Model:', teff, logg, z)
	
	if data is not None:
		order = data.order
	# read in a model
	model    = nsp.Model(teff=teff, logg=logg, z=z, pgs=pgs, modelset=modelset, instrument=instrument, band=band)
	#model    = nsp.Model(teff=teff, logg=logg, feh=z, pgs=pgs, modelset=modelset, instrument=instrument, band=band)
	#print('TEST1', model.flux)
	
	# wavelength offset
	#model.wave += wave_offset

	# apply vsini
	if vsini_set is True:
		model.flux = nsp.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True, gaussian=False)
	#print('TEST2', model.flux)
	
	# apply rv (including the barycentric correction)
	model.wave = rvShift(model.wave, rv=rv)
	
	# apply telluric
	if tell is True:
		model = nsp.applyTelluric(model=model, alpha=alpha, airmass='1.5')
	#print('TEST3', model.flux)

	# Line Spread Function
	if lsf is not None:
		model.flux = nsp.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)
	#print('TEST4', model.flux)
	# add a fringe pattern to the model
	#model.flux *= (1+amp*np.sin(freq*(model.wave-phase)))

	# wavelength offset
	model.wave += wave_offset
	
	# integral resampling
	if data is not None:

		if JHK == 'J':
			model.Jflux = np.array(nsp.integralResample(xh=model.wave, yh=model.flux, xl=data.Jwave))
			model.Jwave = data.Jwave
			model.Jflux *= flux_multiplier
			model.Jflux += flux_offset
			return model

		if JHK == 'H':
			model.Hflux = np.array(nsp.integralResample(xh=model.wave, yh=model.flux, xl=data.Hwave))
			model.Hwave = data.Hwave
			model.Hflux *= flux_multiplier
			model.Hflux += flux_offset
			return model

		if JHK == 'K':
			model.Kflux = np.array(nsp.integralResample(xh=model.wave, yh=model.flux, xl=data.Kwave))
			model.Kwave = data.Kwave
			model.Kflux *= flux_multiplier
			model.Kflux += flux_offset
			return model

		model.flux = np.array(nsp.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
		model.wave = data.wave

		
		# contunuum correction (not for OSIRIS data)
		#model = nsp.continuum(data=data, mdl=model)
		if smooth:
			smoothfluxmed = sp.ndimage.filters.uniform_filter(model.flux, size=200) #replica of IDL

			model.flux -= smoothfluxmed
		
		# flux multiplicate
		model.flux *= flux_multiplier

	# flux offset
	model.flux += flux_offset
	#model.flux **= (1 + flux_exponent_offset)

	return model


def rvShift(wavelength, rv):
	"""
	Perform the radial velocity correction.

	Parameters
	----------
	wavelength 	: 	numpy array 
					model wavelength (in Angstroms)

	rv 			: 	float
					radial velocity shift (in km/s)

	Returns
	-------
	wavelength 	: 	numpy array 
					shifted model wavelength (in Angstroms)
	"""
	return wavelength * ( 1 + rv / 299792.458)


def applyTelluric(model, alpha=1, airmass='1.5'):
	"""
	Apply the telluric model on the science model.

	Parameters
	----------
	model 	:	model object
				BT Settl model
	alpha 	: 	float
				telluric scaling factor (the power on the flux)

	Returns
	-------
	model 	: 	model object
				BT Settl model times the corresponding model

	"""
	# read in a telluric model
	wavelow  = model.wave[0] - 10
	wavehigh = model.wave[-1] + 10

	telluric_model = nsp.getTelluric(wavelow=wavelow, wavehigh=wavehigh, alpha=alpha, airmass=airmass)

	# apply the telluric alpha parameter
	#telluric_model.flux = telluric_model.flux**(alpha)

	#if len(model.wave) > len(telluric_model.wave):
	#	print("The model has a higher resolution ({}) than the telluric model ({})."\
	#		.format(len(model.wave),len(telluric_model.wave)))
	#	model.flux = np.array(nsp.integralResample(xh=model.wave, 
	#		yh=model.flux, xl=telluric_model.wave))
	#	model.wave = telluric_model.wave
	#	model.flux *= telluric_model.flux

	#elif len(model.wave) < len(telluric_model.wave):
	## This should be always true
	telluric_model.flux = np.array(nsp.integralResample(xh=telluric_model.wave, yh=telluric_model.flux, xl=model.wave))

	telluric_model.wave = model.wave

	model.flux *= telluric_model.flux

	#elif len(model.wave) == len(telluric_model.wave):
	#	model.flux *= telluric_model.flux
		
	return model


def convolveTelluric(lsf, telluric_data, alpha=1):
	"""
	Return a convolved telluric transmission model given a telluric data and lsf.
	"""
	# get a telluric standard model
	wavelow               = telluric_data.wave[0]  - 50
	wavehigh              = telluric_data.wave[-1] + 50
	telluric_model        = nsp.getTelluric(wavelow=wavelow,wavehigh=wavehigh)
	telluric_model.flux **= alpha
	# lsf
	telluric_model.flux = nsp.broaden(wave=telluric_model.wave, flux=telluric_model.flux, vbroad=lsf, rotate=False, gaussian=True)
	# resample
	telluric_model.flux = np.array(nsp.integralResample(xh=telluric_model.wave, yh=telluric_model.flux, xl=telluric_data.wave))
	telluric_model.wave = telluric_data.wave
	return telluric_model


def getLSF2(telluric_data, continuum=True, test=False, save_path=None):
	"""
	Return a best LSF value from a telluric data.
	"""
	
	data = copy.deepcopy(telluric_data)

	def bestParams(data, i, alpha, c2, c0):

		data2          = copy.deepcopy(data)
		data2.wave     = data2.wave + c0
		telluric_model = nsp.convolveTelluric(i, data2, alpha=alpha)
		model          = nsp.continuum(data=data2, mdl=telluric_model)
		#plt.figure(2)
		#plt.plot(model.wave, model.flux+c2, 'r-', alpha=0.5)
		#plt.plot(data.wave*c1+c0, data.flux, 'b-', alpha=0.5)
		#plt.close()
		#plt.show()
		#sys.exit()
		return model.flux + c2

	def bestParams2(theta, data):

		i, alpha, c2, c0, c1 = theta 
		data2                = copy.deepcopy(data)
		data2.wave           = data2.wave*c1 + c0
		telluric_model       = nsp.convolveTelluric(i, data2, alpha=alpha)
		model                = nsp.continuum(data=data2, mdl=telluric_model)
		return np.sum(data.flux - (model.flux + c2))**2

	from scipy.optimize import curve_fit, minimize

	popt, pcov = curve_fit(bestParams, data, data.flux, p0=[4.01, 1.01, 0.01, 1.01], maxfev=1000000, epsfcn=0.1)

	#nll = lambda *args: bestParams2(*args)
	#results = minimize(nll, [3., 1., 0.1, -10., 1.], args=(data))
	#popt = results['x']

	data.wave      = data.wave+popt[3]

	telluric_model = nsp.convolveTelluric(popt[0], data, alpha=popt[1])
	model          = nsp.continuum(data=data, mdl=telluric_model)

	#model.flux * np.e**(-popt[2]) + popt[3]
	model.flux + popt[2]

	return popt[0]


def getLSF(telluric_data, alpha=1.0, continuum=True,test=False,save_path=None):
	"""
	Return a best LSF value from a telluric data.
	"""
	lsf_list = []
	test_lsf = np.arange(3.0,13.0,0.1)
	
	data = copy.deepcopy(telluric_data)
	if continuum is True:
		data = nsp.continuumTelluric(data=data)

	data.flux **= alpha
	for i in test_lsf:
		telluric_model = nsp.convolveTelluric(i,data)
		if telluric_data.order == 59:
			telluric_model.flux **= 3
			# mask hydrogen absorption feature
			data2          = copy.deepcopy(data)
			tell_mdl       = copy.deepcopy(telluric_model)
			mask_pixel     = 450
			data2.wave     = data2.wave[mask_pixel:]
			data2.flux     = data2.flux[mask_pixel:]
			data2.noise    = data2.noise[mask_pixel:]
			tell_mdl.wave  = tell_mdl.wave[mask_pixel:]
			tell_mdl.flux  = tell_mdl.flux[mask_pixel:]

			chisquare = nsp.chisquare(data2,tell_mdl)

		else:
			chisquare = nsp.chisquare(data,telluric_model)
		lsf_list.append([chisquare,i])

		if test is True:
			plt.plot(telluric_model.wave,telluric_model.flux+(i-3)*10+1,
				'r-',alpha=0.5)

	if test is True:
		plt.plot(data.wave,data.flux,
			'k-',label='telluric data',alpha=0.5)
		plt.title("Test LSF",fontsize=15)
		plt.xlabel("Wavelength ($\AA$)",fontsize=12)
		plt.ylabel("Transmission + Offset",fontsize=12)
		plt.minorticks_on()
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_lsf_data_mdl.png"\
				.format(data.name, data.order))
		#plt.show()
		plt.close()

		fig, ax = plt.subplots()
		for i in range(len(lsf_list)):
			ax.plot(lsf_list[i][1],lsf_list[i][0],'k.',alpha=0.5)
		ax.plot(min(lsf_list)[1],min(lsf_list)[0],'r.',
			label="best LSF {} km/s".format(min(lsf_list)[1]))
		ax.set_xlabel("LSF (km/s)",fontsize=12)
		ax.set_ylabel("$\chi^2$",fontsize=11)
		plt.minorticks_on()
		plt.legend(fontsize=10)
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_lsf_chi2.png"\
				.format(data.name, data.order))
		#plt.show()
		plt.close()

	lsf = min(lsf_list)[1]

	if telluric_data.order == 61 or telluric_data.order == 62 \
	or telluric_data.order == 63: #or telluric_data.order == 64:
		lsf = 5.5
		print("The LSF is obtained from orders 60 and 65 (5.5 km/s).")

	return lsf


def getAlpha(telluric_data, lsf, continuum=True, test=False, save_path=None):
	"""
	Return a best alpha value from a telluric data.
	"""
	alpha_list = []
	test_alpha = np.arange(0.1,7,0.1)

	data = copy.deepcopy(telluric_data)
	if continuum is True:
		data = nsp.continuumTelluric(data=data, order=data.order)

	for i in test_alpha:
		telluric_model = nsp.convolveTelluric(lsf,data,
			alpha=i)
		#telluric_model.flux **= i 
		if data.order == 59:
			# mask hydrogen absorption feature
			data2          = copy.deepcopy(data)
			tell_mdl       = copy.deepcopy(telluric_model)
			mask_pixel     = 450
			data2.wave     = data2.wave[mask_pixel:]
			data2.flux     = data2.flux[mask_pixel:]
			data2.noise    = data2.noise[mask_pixel:]
			tell_mdl.wave  = tell_mdl.wave[mask_pixel:]
			tell_mdl.flux  = tell_mdl.flux[mask_pixel:]

			chisquare = nsp.chisquare(data2,tell_mdl)

		else:
			chisquare = nsp.chisquare(data,telluric_model)
		alpha_list.append([chisquare,i])

		if test is True:
			plt.plot(telluric_model.wave,telluric_model.flux+i*10,
				'k-',alpha=0.5)

	if test is True:
		plt.plot(telluric_data.wave,telluric_data.flux,
			'r-',alpha=0.5)
		plt.rc('font', family='sans-serif')
		plt.title("Test Alpha",fontsize=15)
		plt.xlabel("Wavelength ($\AA$)",fontsize=12)
		plt.ylabel("Transmission + Offset",fontsize=12)
		plt.minorticks_on()
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_alpha_data_mdl.png"\
				.format(telluric_data.name,
					telluric_data.order))
		plt.show()
		plt.close()

		fig, ax = plt.subplots()
		plt.rc('font', family='sans-serif')
		for i in range(len(alpha_list)):
			ax.plot(alpha_list[i][1],alpha_list[i][0],'k.',alpha=0.5)
		ax.plot(min(alpha_list)[1],min(alpha_list)[0],'r.',
			label="best alpha {}".format(min(alpha_list)[1]))
		ax.set_xlabel(r"$\alpha$",fontsize=12)
		ax.set_ylabel("$\chi^2$",fontsize=12)
		plt.minorticks_on()
		plt.legend(fontsize=10)
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_alpha_chi2.png"\
				.format(telluric_data.name,
					telluric_data.order))
		plt.show()
		plt.close()

	alpha = min(alpha_list)[1]

	return alpha


def getFringeFrequecy(tell_data, test=False):
	"""
	Use the Lomb-Scargle Periodogram to identify 
	the fringe pattern.
	"""
	tell_sp  = copy.deepcopy(tell_data)

	## continuum correction
	tell_sp  = nsp.continuumTelluric(data=tell_sp, order=tell_sp.order)

	## get a telluric model
	lsf      = nsp.getLSF(tell_sp)
	alpha    = nsp.getAlpha(tell_sp,lsf)
	tell_mdl = nsp.convolveTelluric(lsf=lsf,
		telluric_data=tell_sp,alpha=alpha)

	## fit the fringe pattern in the residual
	pgram_x = np.array(tell_sp.wave,float)[10:-10]
	pgram_y = np.array(tell_sp.flux - tell_mdl.flux,float)[10:-10]
	offset  = np.mean(pgram_y)
	pgram_y -= offset
	mask    = np.where(pgram_y - 1.5 * np.absolute(np.std(pgram_y)) > 0)
	pgram_x = np.delete(pgram_x, mask)
	pgram_y = np.delete(pgram_y, mask)
	pgram_x = np.array(pgram_x, float)
	pgram_y = np.array(pgram_y, float)

	#f = np.linspace(0.01,10,100000)
	f = np.linspace(1.0,10,100000)

	## Lomb Scargle Periodogram
	pgram = signal.lombscargle(pgram_x, pgram_y, f)

	if test:
		fig, ax = plt.subplots(figsize=(16,6))
		ax.plot(f,pgram, 'k-', label='residual',alpha=0.5)
		ax.set_xlabel('frequency')
		plt.legend()
		plt.show()
		plt.close()

	return f[np.argmax(pgram)]


def initModelFit(sci_data, lsf, modelset='aces2013'):
	"""
	Conduct simple chisquare fit to obtain the initial parameters
	for the forward modeling MCMC.

	The function would calculate the chisquare for teff, logg, vini, rv, and alpha.

	Parameters
	----------
	data 				:	spectrum object
							input science data

	lsf 				:	float
							line spread function for OSIRIS

	Returns
	-------
	best_params_dic 	:	dic
							a dictionary that stores the best parameters for 
							teff, logg, vsini, rv, and alpha

	chisquare 			:	int
							minimum chisquare

	"""
	data            = copy.deepcopy(sci_data)

	## set up the parameter grid for chisquare computation
	teff_array      = np.arange(1200,3001,100)
	logg_array      = np.arange(3.5,5.51,0.5)
	vsini_array     = np.arange(10,101,10)
	rv_array        = np.arange(-200,201,50)
	alpha_array     = np.arange(0.5,2.01,0.5)
	chisquare_array = np.empty(len(teff_array)*len(logg_array)*len(vsini_array)*len(rv_array)*len(alpha_array))\
	.reshape(len(teff_array),len(logg_array),len(vsini_array),len(rv_array),len(alpha_array))

	time1 = time.time()
	for i, teff in enumerate(teff_array):
		for j, logg in enumerate(logg_array):
			for k, vsini in enumerate(vsini_array):
				for l, rv in enumerate(rv_array):
					for m, alpha in enumerate(alpha_array):
						model = nsp.makeModel(teff, logg, 0.0, vsini, rv, alpha, 0, 0,
							lsf=lsf, order=data.order, data=data, modelset=modelset)
						chisquare_array[i,j,k,l,m] = nsp.chisquare(data, model)
	time2 = time.time()
	print("total time:",time2-time1)

	ind = np.unravel_index(np.argmin(chisquare_array, axis=None), chisquare_array.shape)
	print("ind ",ind)
	chisquare       = chisquare_array[ind]

	best_params_dic = {'teff':teff_array[ind[0]], 'logg':logg_array[ind[1]], 
	'vsini':vsini_array[ind[2]], 'rv':rv_array[ind[3]], 'alpha':alpha_array[ind[4]]}

	print(best_params_dic, chisquare)

	return best_params_dic , chisquare

