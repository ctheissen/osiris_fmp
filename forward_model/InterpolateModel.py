#!/usr/bin/env python
import numpy as np
import sys, os, os.path, time
from astropy.table import Table


################################################################

def InterpModel(Teff, Logg, modelset='aces2013', instrument='OSIRIS', band='Kbb'):

	FULL_PATH  = os.path.realpath(__file__)
	BASE, NAME = os.path.split(FULL_PATH)

	# Check the instrument and band
	if instrument == 'OSIRIS':
		bandname  = '%s-%s-RAW'%(instrument, band)
	if instrument == 'CHARIS':
		bandname  = '%s-%s-RAW'%(instrument, band)

	# Check the model set
	if modelset == 'btsettl08':
		path = BASE + '/../libraries/btsettl08/%s/'%bandname
	elif modelset == 'phoenixaces' :
		path = BASE + '/../libraries/phoenixaces/%s/'%bandname
	elif modelset == 'aces2013' :
		path = BASE + '/../libraries/aces2013/%s/'%bandname
	elif modelset == 'aces-agss-cond-2011' :
		path = BASE + '/../libraries/aces-agss-cond-2011/%s/'%bandname
	elif modelset == 'agss09-dusty' :
		path = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/%s/'%bandname
		

	def bilinear_interpolation(x, y, points):
		'''Interpolate (x,y) from values associated with four points.

		The four points are a list of four triplets:  (x, y, value).
		The four points can be in any order.  They should form a rectangle.

			>>> bilinear_interpolation(12, 5.5,
			...                        [(10, 4, 100),
			...                         (20, 4, 200),
			...                         (10, 6, 150),
			...                         (20, 6, 300)])
			165.0

		'''
		# See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

		#print(points)
		#points = sorted(points, key = lambda x: (x[0]))               # order points by x, then by y
		(x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
		#print(x,y)
		#print(x1.data, y1.data, _x1.data, y2.data, x2.data, _y1.data, _x2.data, _y2.data)
		#print(not x1 <= x <= x2, not y1 <= y <= y2)
		if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
			raise ValueError('points do not form a rectangle')
		if not x1 <= x <= x2 or not y1 <= y <= y2:
			raise ValueError('(x, y) not within the rectangle')

		interpFlux = 10**((q11 * (x2 - x) * (y2 - y) +
						   q21 * (x - x1) * (y2 - y) +
						   q12 * (x2 - x) * (y - y1) +
						   q22 * (x - x1) * (y - y1)
						   ) / ((x2 - x1) * (y2 - y1) + 0.0))

		#print(x,y,x2,y2, q11 * (x2 - x) * (y2 - y))
		#print(x,y,x1,y2, q21 * (x - x1) * (y2 - y))
		#print(x,y,x2,y1, q12 * (x2 - x) * (y - y1))
		#print(x,y,x1,y1, q22 * (x - x1) * (y - y1))
		return interpFlux


	def GetModel(temp, logg, modelset='aces2013', wave=False):
		feh, en, kzz = 0.00, 0.00, 0.0
		if modelset == 'btsettl08':
			filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'phoenixaces':
			filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'aces2013':
			filename = 'aces2013_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'aces-agss-cond-2011':
			filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'agss09-dusty':
			filename = 'AGSS09-Dusty_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(logg) + '_z{0:.2f}'.format(feh) + '_Kzz{0:.1f}'.format(kzz) + '_%s.txt'%bandname

		Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

		if wave:
			return Tab['wave']
		else:
			return Tab['flux']

	if modelset == 'btsettl08':
		Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'
	elif modelset == 'phoenixaces':
		Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'
	elif modelset == 'aces2013':
		Gridfile = BASE + '/../libraries/aces2013/aces2013_gridparams.csv'
	elif modelset == 'aces-agss-cond-2011':
		Gridfile = BASE + '/../libraries/aces-agss-cond-2011/gridparams.csv'
	elif modelset == 'agss09-dusty':
		Gridfile = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/AGSS09-Dusty_gridparams.csv'

	T1 = Table.read(Gridfile)

	# Check if the model already exists (grid point)
	if (Teff, Logg) in zip(T1['Temp'], T1['Logg']): 
		flux2  = GetModel(T1['Temp'][np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == Teff) & (T1['Logg'] == Logg))])
		waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == Teff) & (T1['Logg'] == Logg))], wave=True)
		return waves2, flux2

	#print(Teff, Logg, x1, x2, y1, y2)
	x0 = T1['Temp'][np.where(T1['Temp'] <= Teff)][-1]
	x1 = T1['Temp'][np.where(T1['Temp'] >= Teff)][0]
	y0 = list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )]))[-1]
	y1 = list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )]))[0]
	

	# Check if the gridpoint exists within the model ranges
	for x in [x0, x1]:
		for y in [y0, y1]:
			if (x, y) not in zip(T1['Temp'], T1['Logg']):
				print('No Model', x, y)
				return 1
	'''
	print(np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1)))
	print(np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2)))
	print(np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1)))
	print(np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2)))
	print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]))
	print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))]))
	print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))]))
	print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))]))
	'''
	# Get the four points
	Points =  [ [np.log10(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))]), T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], modelset=modelset))],
				[np.log10(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y1))], modelset=modelset))],
				[np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y0))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y0))], modelset=modelset))],
				[np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], modelset=modelset))],
			  ]
	#print(Points)
	waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], wave=True, modelset=modelset)

	return waves2, bilinear_interpolation(np.log10(Teff), Logg, Points)



#########################################################################################################



def InterpModel_Log(LogTeff, Logg, modelset='aces2013', instrument='OSIRIS', band='Kbb'):

	FULL_PATH  = os.path.realpath(__file__)
	BASE, NAME = os.path.split(FULL_PATH)

	# Check the instrument and band
	if instrument == 'OSIRIS':
		bandname  = '%s-%s-RAW'%(instrument, band)
	if instrument == 'CHARIS':
		bandname  = '%s-%s-RAW'%(instrument, band)

	# Check the model set
	if modelset == 'btsettl08':
		path = BASE + '/../libraries/btsettl08/%s/'%bandname
	elif modelset == 'phoenixaces' :
		path = BASE + '/../libraries/phoenixaces/%s/'%bandname
	elif modelset == 'aces2013' :
		path = BASE + '/../libraries/aces2013/%s/'%bandname
	elif modelset == 'aces-agss-cond-2011' :
		path = BASE + '/../libraries/aces-agss-cond-2011/%s/'%bandname
	elif modelset == 'agss09-dusty' :
		path = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/%s/'%bandname
		

	def bilinear_interpolation(x, y, points):
		'''Interpolate (x,y) from values associated with four points.

		The four points are a list of four triplets:  (x, y, value).
		The four points can be in any order.  They should form a rectangle.

			>>> bilinear_interpolation(12, 5.5,
			...                        [(10, 4, 100),
			...                         (20, 4, 200),
			...                         (10, 6, 150),
			...                         (20, 6, 300)])
			165.0

		'''
		# See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

		#print(points)
		#points = sorted(points, key = lambda x: (x[0]))               # order points by x, then by y
		(x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
		#print(x,y)
		#print(x1.data, y1.data, _x1.data, y2.data, x2.data, _y1.data, _x2.data, _y2.data)
		#print(not x1 <= x <= x2, not y1 <= y <= y2)
		if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
			raise ValueError('points do not form a rectangle')
		if not x1 <= x <= x2 or not y1 <= y <= y2:
			raise ValueError('(x, y) not within the rectangle')

		interpFlux = 10**((q11 * (x2 - x) * (y2 - y) +
						   q21 * (x - x1) * (y2 - y) +
						   q12 * (x2 - x) * (y - y1) +
						   q22 * (x - x1) * (y - y1)
						   ) / ((x2 - x1) * (y2 - y1) + 0.0))

		#print(x,y,x2,y2, q11 * (x2 - x) * (y2 - y))
		#print(x,y,x1,y2, q21 * (x - x1) * (y2 - y))
		#print(x,y,x2,y1, q12 * (x2 - x) * (y - y1))
		#print(x,y,x1,y1, q22 * (x - x1) * (y - y1))
		return interpFlux


	def GetModel(temp, logg, modelset='aces2013', wave=False):
		feh, en, kzz = 0.00, 0.00, 0.0
		#print(int(temp.data[0]), logg)
		if modelset == 'btsettl08':
			filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'phoenixaces':
			filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'aces2013':
			filename = 'aces2013_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'aces-agss-cond-2011':
			filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
		elif modelset == 'agss09-dusty':
			filename = 'AGSS09-Dusty_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_Kzz{0:.1f}'.format(float(kzz)) + '_%s.txt'%bandname

		Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

		if wave:
			return Tab['wave']
		else:
			return Tab['flux']

	if modelset == 'btsettl08':
		Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'
	elif modelset == 'phoenixaces':
		Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'
	elif modelset == 'aces2013':
		Gridfile = BASE + '/../libraries/aces2013/aces2013_gridparams.csv'
	elif modelset == 'aces-agss-cond-2011':
		Gridfile = BASE + '/../libraries/aces-agss-cond-2011/gridparams.csv'
	elif modelset == 'agss09-dusty':
		Gridfile = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/AGSS09-Dusty_gridparams.csv'
	
	T1 = Table.read(Gridfile)

	# Check if the model already exists (grid point)
	if (LogTeff, Logg) in zip(T1['Temp'], T1['Logg']): 
		flux2  = GetModel(T1['Temp'][np.where( (T1['Temp'] == 10**LogTeff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == 10**LogTeff) & (T1['Logg'] == Logg))])
		waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == 10**LogTeff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == 10**LogTeff) & (T1['Logg'] == Logg))], wave=True)
		return waves2, flux2

	#print(10**LogTeff)
	#print(T1['Temp'][np.where(T1['Temp'] <= 10**LogTeff)].data)
	x0 = T1['Temp'][np.where(T1['Temp'] <= 10**LogTeff)][-1]
	x1 = T1['Temp'][np.where(T1['Temp'] >= 10**LogTeff)][0]
	y0 = list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )]))[-1]
	y1 = list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )]))[0]
	

	# Check if the gridpoint exists within the model ranges
	for x in [x0, x1]:
		for y in [y0, y1]:
			if (x, y) not in zip(T1['Temp'], T1['Logg']):
				print('No Model', x, y)
				return 1

	# Get the four points
	Points =  [ [np.log10(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))]), T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], modelset=modelset))],
				[np.log10(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y1))], modelset=modelset))],
				[np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y0))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y0))], modelset=modelset))],
				[np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], modelset=modelset))],
			  ]
	#print(Points)
	waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], wave=True, modelset=modelset)

	return waves2, bilinear_interpolation(LogTeff, Logg, Points)

