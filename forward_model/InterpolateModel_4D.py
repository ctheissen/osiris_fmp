#!/usr/bin/env python
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from numpy.linalg import inv, det


################################################################

def InterpModel_4D(Teff, Logg, PGS, GS, modelset='aces-pso318', instrument='OSIRIS', band='Kbb'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the instrument and band
    if instrument == 'OSIRIS':
        bandname  = '%s-%s-RAW'%(instrument, band)

    # Check the model set
    if modelset == 'aces-pso318':
        path = BASE + '/../libraries/aces-pso318/%s/'%bandname
    else:
        raise ValueError('Only aces-pso318 modelset available for 4D interpolation')
        

    def quadlinear_interpolation(x, y, z, t, points):
        '''Interpolate (x,y) from values associated with 16 points.

        Custom routine

        '''

        (x0, y0, z0, t0, q0000), (x1, y0, z0, t0, q1000), (x0, y1, z0, t0, q0100), (x0, y0, z1, t0, q0010), (x0, y0, z0, t1, q0001), \
        (x1, y0, z0, t1, q1001), (x0, y1, z0, t1, q0101), (x0, y0, z1, t1, q0011), (x1, y0, z1, t1, q1011), (x0, y1, z1, t1, q0111), \
        (x1, y1, z1, t1, q1111), (x0, y1, z1, t0, q0110), (x1, y0, z1, t0, q1010), (x1, y1, z0, t0, q1100), (x1, y1, z0, t1, q1101), \
        (x1, y1, z1, t0, q1110) = points
        x0 = x0.data[0]
        x1 = x1.data[0]
        y0 = y0.data[0]
        y1 = y1.data[0]
        z0 = z0.data[0]
        z1 = z1.data[0]
        t0 = t0.data[0]
        t1 = t1.data[0]
        t1 = 2
        print(x0,x1,y0,y1,z0,z1,t0,t1)

        #print(x, y, x0.data, y0.data, _x0.data, y1.data, x1.data, _y0.data, _x1.data, _y1.data)
        #print(not x1 <= x <= x2, not y1 <= y <= y2)
        #if x0 != _x0 or x1 != _x1 or y0 != _y0 or y1 != _y1:
        #    raise ValueError('points do not form a rectangle')
        #if not x0 <= x <= x1 or not y0 <= y <= y1:
        #    raise ValueError('(x, y) not within the rectangle')

        c = np.array([ [1., x0, y0, z0, t0, x0*y0, x0*z0, x0*t0, y0*z0, y0*t0, z0*t0, x0*y0*z0, x0*y0*t0, x0*z0*t0, y0*z0*t0, x0*y0*z0*t0], #0000
                       [1., x1, y0, z0, t0, x1*y0, x1*z0, x1*t0, y0*z0, y0*t0, z0*t0, x1*y0*z0, x1*y0*t0, x1*z0*t0, y0*z0*t0, x1*y0*z0*t0], #1000
                       [1., x0, y1, z0, t0, x0*y1, x0*z0, x0*t0, y1*z0, y1*t0, z0*t0, x0*y1*z0, x0*y1*t0, x0*z0*t0, y1*z0*t0, x0*y1*z0*t0], #0100
                       [1., x0, y0, z1, t0, x0*y0, x0*z1, x0*t0, y0*z1, y0*t0, z1*t0, x0*y0*z1, x0*y0*t0, x0*z1*t0, y0*z1*t0, x0*y0*z1*t0], #0010
                       [1., x0, y0, z0, t1, x0*y0, x0*z0, x0*t1, y0*z0, y0*t1, z0*t1, x0*y0*z0, x0*y0*t1, x0*z0*t1, y0*z0*t1, x0*y0*z0*t1], #0001
                       [1., x1, y0, z0, t1, x1*y0, x1*z0, x1*t1, y0*z0, y0*t1, z0*t1, x1*y0*z0, x1*y0*t1, x1*z0*t1, y0*z0*t1, x1*y0*z0*t1], #1001
                       [1., x0, y1, z0, t1, x0*y1, x0*z0, x0*t1, y1*z0, y1*t1, z0*t1, x0*y1*z0, x0*y1*t1, x0*z0*t1, y1*z0*t1, x0*y1*z0*t1], #0101
                       [1., x0, y0, z1, t1, x0*y0, x0*z1, x0*t1, y0*z1, y0*t1, z1*t1, x0*y0*z1, x0*y0*t1, x0*z1*t1, y0*z1*t1, x0*y0*z1*t1], #0011
                       [1., x1, y0, z1, t1, x1*y0, x1*z1, x1*t1, y0*z1, y0*t1, z1*t1, x1*y0*z1, x1*y0*t1, x1*z1*t1, y0*z1*t1, x1*y0*z1*t1], #1011
                       [1., x0, y1, z1, t1, x0*y1, x0*z1, x0*t1, y1*z1, y1*t1, z1*t1, x0*y1*z1, x0*y1*t1, x0*z1*t1, y1*z1*t1, x0*y1*z1*t1], #0111
                       [1., x1, y1, z1, t1, x1*y1, x1*z1, x1*t1, y1*z1, y1*t1, z1*t1, x1*y1*z1, x1*y1*t1, x1*z1*t1, y1*z1*t1, x1*y1*z1*t1], #1111
                       [1., x0, y1, z1, t0, x0*y1, x0*z1, x0*t0, y1*z1, y1*t0, z1*t0, x0*y1*z1, x0*y1*t0, x0*z1*t0, y1*z1*t0, x0*y1*z1*t0], #0110
                       [1., x1, y0, z1, t0, x1*y0, x1*z1, x1*t0, y0*z1, y0*t0, z1*t0, x1*y0*z1, x1*y0*t0, x1*z1*t0, y0*z1*t0, x1*y0*z1*t0], #1010
                       [1., x1, y1, z0, t0, x1*y1, x1*z0, x1*t0, y1*z0, y1*t0, z0*t0, x1*y1*z0, x1*y1*t0, x1*z0*t0, y1*z0*t0, x1*y1*z0*t0], #1100
                       [1., x1, y1, z0, t1, x1*y1, x1*z0, x1*t1, y1*z0, y1*t1, z0*t1, x1*y1*z0, x1*y1*t1, x1*z0*t1, y1*z0*t1, x1*y1*z0*t1], #1101
                       [1., x1, y1, z1, t0, x1*y1, x1*z1, x1*t0, y1*z1, y1*t0, z1*t0, x1*y1*z1, x1*y1*t0, x1*z1*t0, y1*z1*t0, x1*y1*z1*t0], #1110
                      ], dtype='float')
        print(c)
        print(det(c))
        invc      = inv(c)
        transinvc = np.transpose(invc)

        final = np.dot(transinvc, [1, x, y, z, t, x*y, x*z, x*t, y*z, y*t, z*t, x*y*z, x*y*t, x*z*t, y*z*t, x*y*z*t])
        print('Final Sum:', np.sum(final))


        interpFlux = 10**( (q0000*final[0] + q1000*final[1] + q0100*final[2] + q0010*final[3] + q0001*final[4] +
                            q1001*final[5] + q0101*final[6] + q0011*final[7] + q1011*final[8] + q0111*final[9] +
                            q1111*final[10]+ q0110*final[11]+ q1010*final[12]+ q1100*final[13]+ q1101*final[14]+
                            q1110*final[15])
                           )

        #print(x,y,x2,y2, q11 * (x2 - x) * (y2 - y))
        #print(x,y,x1,y2, q21 * (x - x1) * (y2 - y))
        #print(x,y,x2,y1, q12 * (x2 - x) * (y - y1))
        #print(x,y,x1,y1, q22 * (x - x1) * (y - y1))
        #print('b11', (x2.data - x) * (y2.data - y) / ((x2.data - x1.data) * (y2.data - y1.data)))
        #print('b12', (x - x1.data) * (y2.data - y) / ((x2.data - x1.data) * (y2.data - y1.data)))
        #print('b21', (x2.data - x) * (y - y1.data) / ((x2.data - x1.data) * (y2.data - y1.data)))
        #print('b22', (x - x1.data) * (y - y1.data) / ((x2.data - x1.data) * (y2.data - y1.data)))
        #print('b11', 10**((x2.data - x) * (y2.data - y) / (x2.data - x1.data) * (y2.data - y1.data)))
        #print('b12', 10**((x - x1.data) * (y2.data - y) / (x2.data - x1.data) * (y2.data - y1.data)))
        #print('b21', 10**((x2.data - x) * (y - y1.data) / (x2.data - x1.data) * (y2.data - y1.data)))
        #print('b22', 10**((x - x1.data) * (y - y1.data) / (x2.data - x1.data) * (y2.data - y1.data)))
        return interpFlux


    def GetModel(temp, logg, pgs, gs, modelset='aces-pso318', wave=False):
        kzz = int(1e8)
        #print(temp.data[0], logg.data[0], pgs.data[0], gs.data[0])
        if modelset == 'aces-pso318':
            filename = 'aces-pso318_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg.data[0])) + '_pgs' + '{}'.format(pgs.data[0]) + '_Kzz' + '{}'.format(kzz) + '_gs' + '{0:.1f}'.format(gs.data[0]) + '_%s.txt'%bandname
            #print('File', filename)
        else:
            raise ValueError('Only aces-pso318 modelset available for 4D interpolation')
        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    def myround(x, base=.5):
        return base * round(float(x)/base)

    def findlogg(logg):
        LoggArr = np.arange(2.5, 6, 0.5)
        dist    = (LoggArr - logg)**2
        return LoggArr[np.argsort(dist)][0:2]

    if modelset == 'aces-pso318':
        Gridfile = BASE + '/../libraries/aces-pso318/aces-pso318_gridparams.csv'
    T0 = Table.read(Gridfile)
    T1 = T0[np.where(T0['Kzz'] == 1e8)] # not using Kzz yet!

    # Check if the model already exists (grid point)
    if (Teff, Logg, PGS, GS) in zip(T1['Temp'], T1['Logg'], T1['pgs'], T1['gs']): 
        index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) & (T1['pgs'] == PGS) & (T1['gs'] == GS) )
        flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['pgs'][index0], T1['gs'][index0] )
        waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['pgs'][index0], T1['gs'][index0], wave=True)
        return waves2, flux2

    #x1     = np.floor(Teff/100.)*100
    #x2     = np.ceil(Teff/100.)*100
    #print('1', x1, x2)
    #y0, y1 = sorted(findlogg(Logg))

    # Get the nearest models to the gridpoint (Temp)
    #print(T1['Temp'][np.where(T1['Temp'] <= x1)])
    #print(T1['Temp'][np.where(T1['Temp'] >= x2)])
    x0 = T1['Temp'][np.where(T1['Temp'] <= Teff)][-1]
    x1 = T1['Temp'][np.where(T1['Temp'] >= Teff)][0]
    print(x0, Teff, x1)
    y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
    y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
    print(y0, Logg, y1)
    z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
    #print(z0, PGS, z0 <= PGS, z0 >= PGS)
    z1 = sorted(T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )])[0]
    print(z0, PGS, z1)
    t0 = sorted(T1['gs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & 
                                   ( (T1['pgs'] == z0) | (T1['pgs'] == z1) ) & (T1['gs'] <= GS))])[-1]
    t1 = sorted(T1['gs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & 
                                   ( (T1['pgs'] == z0) | (T1['pgs'] == z1) ) & (T1['gs'] >= GS))])[0]
    print(t0, GS, t1)

    # Check if the gridpoint exists within the model ranges
    for x in [x0, x1]:
        for y in [y0, y1]:
            for z in [z0, z1]:
                for t in [t0, t1]:
                    if (x, y, z, t) not in zip(T1['Temp'], T1['Logg'], T1['pgs'], T1['gs']):
                        print('No Model', x, y, z, t)
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
    # Get the 16 points
    ind0000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['pgs'] == z0) & (T1['gs'] == t0) ) # 0000
    ind1000 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['pgs'] == z0) & (T1['gs'] == t0) ) # 1000
    ind0100 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['pgs'] == z0) & (T1['gs'] == t0) ) # 0100
    ind0010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['pgs'] == z1) & (T1['gs'] == t0) ) # 0010
    ind0001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['pgs'] == z0) & (T1['gs'] == t1) ) # 0001
    ind1001 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['pgs'] == z0) & (T1['gs'] == t1) ) # 1001
    ind0101 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['pgs'] == z0) & (T1['gs'] == t1) ) # 0101
    ind0011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['pgs'] == z1) & (T1['gs'] == t1) ) # 0011
    ind1011 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['pgs'] == z1) & (T1['gs'] == t1) ) # 1011
    ind0111 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['pgs'] == z1) & (T1['gs'] == t1) ) # 0111
    ind1111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] == z1) & (T1['gs'] == t1) ) # 1111
    ind0110 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['pgs'] == z1) & (T1['gs'] == t0) ) # 0110
    ind1010 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['pgs'] == z1) & (T1['gs'] == t0) ) # 1010
    ind1100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] == z0) & (T1['gs'] == t0) ) # 1100
    ind1101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] == z0) & (T1['gs'] == t1) ) # 1101
    ind1110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] == z1) & (T1['gs'] == t0) ) # 1110
    Points =  [ [np.log10(T1['Temp'][ind0000]), T1['Logg'][ind0000], np.log10(T1['pgs'][ind0000]), T1['gs'][ind0000], 
                 np.log10(GetModel(T1['Temp'][ind0000], T1['Logg'][ind0000], T1['pgs'][ind0000], T1['gs'][ind0000], modelset=modelset))],
                [np.log10(T1['Temp'][ind1000]), T1['Logg'][ind1000], np.log10(T1['pgs'][ind1000]), T1['gs'][ind1000], 
                 np.log10(GetModel(T1['Temp'][ind1000], T1['Logg'][ind1000], T1['pgs'][ind1000], T1['gs'][ind1000], modelset=modelset))],
                [np.log10(T1['Temp'][ind0100]), T1['Logg'][ind0100], np.log10(T1['pgs'][ind0100]), T1['gs'][ind0100], 
                 np.log10(GetModel(T1['Temp'][ind0100], T1['Logg'][ind0100], T1['pgs'][ind0100], T1['gs'][ind0100], modelset=modelset))],
                [np.log10(T1['Temp'][ind0010]), T1['Logg'][ind0010], np.log10(T1['pgs'][ind0010]), T1['gs'][ind0010], 
                 np.log10(GetModel(T1['Temp'][ind0010], T1['Logg'][ind0010], T1['pgs'][ind0010], T1['gs'][ind0010], modelset=modelset))],
                [np.log10(T1['Temp'][ind0001]), T1['Logg'][ind0001], np.log10(T1['pgs'][ind0001]), T1['gs'][ind0001], 
                 np.log10(GetModel(T1['Temp'][ind0001], T1['Logg'][ind0001], T1['pgs'][ind0001], T1['gs'][ind0001], modelset=modelset))],
                [np.log10(T1['Temp'][ind1001]), T1['Logg'][ind1001], np.log10(T1['pgs'][ind1001]), T1['gs'][ind1001], 
                 np.log10(GetModel(T1['Temp'][ind1001], T1['Logg'][ind1001], T1['pgs'][ind1001], T1['gs'][ind1001], modelset=modelset))],
                [np.log10(T1['Temp'][ind0101]), T1['Logg'][ind0101], np.log10(T1['pgs'][ind0101]), T1['gs'][ind0101], 
                 np.log10(GetModel(T1['Temp'][ind0101], T1['Logg'][ind0101], T1['pgs'][ind0101], T1['gs'][ind0101], modelset=modelset))],
                [np.log10(T1['Temp'][ind0011]), T1['Logg'][ind0011], np.log10(T1['pgs'][ind0011]), T1['gs'][ind0011], 
                 np.log10(GetModel(T1['Temp'][ind0011], T1['Logg'][ind0011], T1['pgs'][ind0011], T1['gs'][ind0011], modelset=modelset))],
                [np.log10(T1['Temp'][ind1011]), T1['Logg'][ind1011], np.log10(T1['pgs'][ind1011]), T1['gs'][ind1011], 
                 np.log10(GetModel(T1['Temp'][ind1011], T1['Logg'][ind1011], T1['pgs'][ind1011], T1['gs'][ind1011], modelset=modelset))],
                [np.log10(T1['Temp'][ind0111]), T1['Logg'][ind0111], np.log10(T1['pgs'][ind0111]), T1['gs'][ind0111], 
                 np.log10(GetModel(T1['Temp'][ind0111], T1['Logg'][ind0111], T1['pgs'][ind0111], T1['gs'][ind0111], modelset=modelset))],
                [np.log10(T1['Temp'][ind1111]), T1['Logg'][ind1111], np.log10(T1['pgs'][ind1111]), T1['gs'][ind1111], 
                 np.log10(GetModel(T1['Temp'][ind1111], T1['Logg'][ind1111], T1['pgs'][ind1111], T1['gs'][ind1111], modelset=modelset))],
                [np.log10(T1['Temp'][ind0110]), T1['Logg'][ind0110], np.log10(T1['pgs'][ind0110]), T1['gs'][ind0110], 
                 np.log10(GetModel(T1['Temp'][ind0110], T1['Logg'][ind0110], T1['pgs'][ind0110], T1['gs'][ind0110], modelset=modelset))],
                [np.log10(T1['Temp'][ind1010]), T1['Logg'][ind1010], np.log10(T1['pgs'][ind1010]), T1['gs'][ind1010], 
                 np.log10(GetModel(T1['Temp'][ind1010], T1['Logg'][ind1010], T1['pgs'][ind1010], T1['gs'][ind1010], modelset=modelset))],
                [np.log10(T1['Temp'][ind1100]), T1['Logg'][ind1100], np.log10(T1['pgs'][ind1100]), T1['gs'][ind1100], 
                 np.log10(GetModel(T1['Temp'][ind1100], T1['Logg'][ind1100], T1['pgs'][ind1100], T1['gs'][ind1100], modelset=modelset))],
                [np.log10(T1['Temp'][ind1101]), T1['Logg'][ind1101], np.log10(T1['pgs'][ind1101]), T1['gs'][ind1101], 
                 np.log10(GetModel(T1['Temp'][ind1101], T1['Logg'][ind1101], T1['pgs'][ind1101], T1['gs'][ind1101], modelset=modelset))],
                [np.log10(T1['Temp'][ind1110]), T1['Logg'][ind1110], np.log10(T1['pgs'][ind1110]), T1['gs'][ind1110], 
                 np.log10(GetModel(T1['Temp'][ind1110], T1['Logg'][ind1110], T1['pgs'][ind1110], T1['gs'][ind1110], modelset=modelset))],
              ]
    #print(Points)
    waves2 = GetModel(T1['Temp'][ind1111], T1['Logg'][ind1111], T1['pgs'][ind1111], T1['gs'][ind1111], wave=True, modelset=modelset)

    return waves2, quadlinear_interpolation(np.log10(Teff), Logg, np.log10(PGS), GS, Points)

