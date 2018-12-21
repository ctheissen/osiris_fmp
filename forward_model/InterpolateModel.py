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

    # Check the model set
    if modelset == 'btsettl08':
        path = BASE + '/../libraries/btsettl08/%s/'%bandname
    elif modelset == 'phoenixaces' :
        path = BASE + '/../libraries/phoenixaces/%s/'%bandname
    elif modelset == 'aces2013' :
        path = BASE + '/../libraries/aces2013/%s/'%bandname
        

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

        points = sorted(points)               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return 10**((q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0))


    def GetModel(temp, logg, modelset='aces2013', wave=False):
        feh, en = 0.00, 0.00
        if modelset == 'btsettl08':
            filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
        elif modelset == 'phoenixaces':
            filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname
        elif modelset == 'aces2013':
            filename = 'aces2013_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_%s.txt'%bandname

        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    def myround(x, base=.5):
        return base * round(float(x)/base)

    def findlogg(logg):
        LoggArr = np.arange(2.5, 6, 0.5)
        dist = (LoggArr - logg)**2
        return LoggArr[np.argsort(dist)][0:2]

    if modelset == 'btsettl08':
        Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'
    elif modelset == 'phoenixaces':
        Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'
    elif modelset == 'aces2013':
        Gridfile = BASE + '/../libraries/aces2013/aces2013_gridparams.csv'
    T1 = Table.read(Gridfile)

    # Check if the model already exists (grid point)
    if (Teff, Logg) in zip(T1['Temp'], T1['Logg']): 
        flux2  = GetModel(T1['Temp'][np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == Teff) & (T1['Logg'] == Logg))])
        waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == Teff) & (T1['Logg'] == Logg))], wave=True)
        return waves2, flux2

    x1     = np.floor(Teff/100.)*100
    x2     = np.ceil(Teff/100.)*100
    y1, y2 = findlogg(Logg)

    # Get the nearest models to the gridpoint (Temp)
    x1 = T1['Temp'][np.where(T1['Temp'] <= x1)][-1]
    x2 = T1['Temp'][np.where(T1['Temp'] >= x2)][0]

    # Check if the gridpoint exists within the model ranges
    for x in [x1, x2]:
        for y in [y1, y2]:
            if (x, y) not in zip(T1['Temp'], T1['Logg']):
                print('No Model', x, y)
                return 1
    
    # Get the four points
    Points =  [ [np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y2))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y2))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y1))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))]), T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y2))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))], T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y2))], modelset=modelset))],
              ]

    waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], wave=True, modelset=modelset)

    return waves2, bilinear_interpolation(np.log10(Teff), Logg, Points)

