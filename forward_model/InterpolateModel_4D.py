import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from numpy.linalg import inv, det
from numpy.linalg import inv, det
import osiris_fmp as ospf


#############

def GetModel(temp, logg, pgs, gs, modelset='aces-pso318', wave=False, bandname=None):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    #print(Teff, Logg, PGS, GS)

    # Check the instrument and band
    #bandname = '%s-%s-RAW'%(instrument.upper(), band.upper())

    # Check the model set
    if modelset.lower() == 'aces-pso318':
        path = BASE + '/../libraries/aces-pso318/%s/'%bandname
    if modelset.lower() == 'vhs1256-pso':
        path = BASE + '/../libraries/VHS1256_PSO/%s/'%bandname
    else:
        raise ValueError('Only [aces-pso318, vhs1256-pso] modelsets available for 4D interpolation')
    kzz = int(1e8)
    #print(temp.data[0], logg.data[0], pgs.data[0], gs.data[0])
    if modelset.lower() == 'aces-pso318':
        filename = 'aces-pso318_t'+ str(int(temp)) + '_g{0:.2f}'.format(float(logg)) + '_pgs{}'.format(pgs) + '_kzz{0:.2f}'.format(kzz) + '_gs{0:.1f}'.format(gs) + '_%s.txt'%bandname
        #print('File', filename)        if modelset.lower() == 'aces-pso318':
    elif modelset.lower() == 'vhs1256-pso':
        metal = 0
        alpha = 0
        filename = 'VHS1256_PSO_t'+ str(int(temp)) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(metal) + '_alpha{0:.2f}'.format(alpha) + '_pgs{0:.2f}'.format(pgs) + '_gs{0:.2f}'.format(gs) + '_kzz{0:.2f}'.format(kzz) + '_%s.txt'%bandname
        #print('File', filename)
    else:
        raise ValueError('Only [aces-pso318, vhs1256-pso] modelsets available for 4D interpolation')
    Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

    if wave:
        return Tab['wave']
    else:
        return Tab['flux']

################################################################

def InterpModel_4D(Teff, Logg, PGS, GS, modelset='aces-pso318', instrument='OSIRIS', band='Kbb'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    #print(Teff, Logg, PGS, GS)

    # Check the instrument and band
    bandname = '%s-%s-RAW'%(instrument.upper(), band.upper())

    # Check the model set
    if modelset.lower() == 'aces-pso318':
        path = BASE + '/../libraries/aces-pso318/%s/'%bandname
    if modelset.lower() == 'vhs1256-pso':
        path = BASE + '/../libraries/VHS1256_PSO/%s/'%bandname
    else:
        raise ValueError('Only aces-pso318 modelset available for 4D interpolation')
        


    def GetModel(temp, logg, pgs, gs, modelset='aces-pso318', wave=False):
        kzz = int(1e8)
        #print(temp.data[0], logg.data[0], pgs.data[0], gs.data[0])
        if modelset.lower() == 'aces-pso318':
            filename = 'aces-pso318_t'+ str(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg.data[0])) + '_pgs{}'.format(pgs.data[0]) + '_kzz{0:.2f}'.format(kzz) + '_gs{0:.1f}'.format(gs.data[0]) + '_%s.txt'%bandname
            #print('File', filename)        if modelset.lower() == 'aces-pso318':
        elif modelset.lower() == 'vhs1256-pso':
            metal = 0
            alpha = 0
            filename = 'VHS1256_PSO_t'+ str(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg.data[0])) + '_z{0:.2f}'.format(metal) + '_alpha{0:.2f}'.format(alpha) + '_pgs{0:.2f}'.format(pgs.data[0]) + '_gs{0:.2f}'.format(gs.data[0]) + '_kzz{0:.2f}'.format(kzz) + '_%s.txt'%bandname
            #print('File', filename)
        else:
            raise ValueError('Only [aces-pso318, vhs1256-pso] modelsets available for 4D interpolation')
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

    if modelset.lower() == 'aces-pso318':
        Gridfile = BASE + '/../libraries/aces-pso318/aces-pso318_gridparams.csv'
        T0 = Table.read(Gridfile, comment='#')
        T1 = T0[np.where(T0['Kzz'] == 1e8)] # not using Kzz yet!
    elif modelset.lower() == 'vhs1256-pso':
        Gridfile = BASE + '/../libraries/VHS1256_PSO/VHS1256_PSO_gridparams.csv'
        T0 = Table.read(Gridfile, comment='#')
        T1 = T0[np.where( (T0['Kzz'] == 1e8) & (T0['PGS'] != 200000) & (T0['PGS'] != 300000) )] # not using Kzz yet!

    # Check if the model already exists (grid point)
    if (Teff, Logg, PGS, GS) in zip(T1['Temp'], T1['Logg'], T1['PGS'], T1['GS']): 
        index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) & (T1['PGS'] == PGS) & (T1['GS'] == GS) )
        flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['PGS'][index0], T1['GS'][index0] )
        waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['PGS'][index0], T1['GS'][index0], wave=True)
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
    #print(x0, Teff, x1)
    y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
    y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
    #print(y0, Logg, y1)
    #print(PGS, GS)
    z0 = T1['PGS'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['PGS'] <= PGS) )][-1]
    #(z0, PGS, z0 <= PGS, z0 >= PGS)
    z1 = sorted(T1['PGS'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['PGS'] >= PGS) )])[0]
    #print(z0, PGS, z1)
    t0 = sorted(T1['GS'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & 
                                   ( (T1['PGS'] == z0) | (T1['PGS'] == z1) ) & (T1['GS'] <= GS))])[-1]
    t1 = sorted(T1['GS'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & 
                                   ( (T1['PGS'] == z0) | (T1['PGS'] == z1) ) & (T1['GS'] >= GS))])[0]
    #print(t0, GS, t1)

    # Check if the gridpoint exists within the model ranges
    for x in [x0, x1]:
        for y in [y0, y1]:
            for z in [z0, z1]:
                for t in [t0, t1]:
                    if (x, y, z, t) not in zip(T1['Temp'], T1['Logg'], T1['PGS'], T1['GS']):
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
    ind0000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['PGS'] == z0) & (T1['GS'] == t0) ) # 0000
    ind1000 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['PGS'] == z0) & (T1['GS'] == t0) ) # 1000
    ind0100 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['PGS'] == z0) & (T1['GS'] == t0) ) # 0100
    ind0010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['PGS'] == z1) & (T1['GS'] == t0) ) # 0010
    ind0001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['PGS'] == z0) & (T1['GS'] == t1) ) # 0001
    ind1001 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['PGS'] == z0) & (T1['GS'] == t1) ) # 1001
    ind0101 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['PGS'] == z0) & (T1['GS'] == t1) ) # 0101
    ind0011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['PGS'] == z1) & (T1['GS'] == t1) ) # 0011
    ind1011 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['PGS'] == z1) & (T1['GS'] == t1) ) # 1011
    ind0111 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['PGS'] == z1) & (T1['GS'] == t1) ) # 0111
    ind1111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] == z1) & (T1['GS'] == t1) ) # 1111
    ind0110 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['PGS'] == z1) & (T1['GS'] == t0) ) # 0110
    ind1010 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['PGS'] == z1) & (T1['GS'] == t0) ) # 1010
    ind1100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] == z0) & (T1['GS'] == t0) ) # 1100
    ind1101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] == z0) & (T1['GS'] == t1) ) # 1101
    ind1110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] == z1) & (T1['GS'] == t0) ) # 1110
    Points =  [ [np.log10(T1['Temp'][ind0000]), T1['Logg'][ind0000], np.log10(T1['PGS'][ind0000]), T1['GS'][ind0000], 
                 np.log10(GetModel(T1['Temp'][ind0000], T1['Logg'][ind0000], T1['PGS'][ind0000], T1['GS'][ind0000], modelset=modelset))],
                [np.log10(T1['Temp'][ind1000]), T1['Logg'][ind1000], np.log10(T1['PGS'][ind1000]), T1['GS'][ind1000], 
                 np.log10(GetModel(T1['Temp'][ind1000], T1['Logg'][ind1000], T1['PGS'][ind1000], T1['GS'][ind1000], modelset=modelset))],
                [np.log10(T1['Temp'][ind0100]), T1['Logg'][ind0100], np.log10(T1['PGS'][ind0100]), T1['GS'][ind0100], 
                 np.log10(GetModel(T1['Temp'][ind0100], T1['Logg'][ind0100], T1['PGS'][ind0100], T1['GS'][ind0100], modelset=modelset))],
                [np.log10(T1['Temp'][ind0010]), T1['Logg'][ind0010], np.log10(T1['PGS'][ind0010]), T1['GS'][ind0010], 
                 np.log10(GetModel(T1['Temp'][ind0010], T1['Logg'][ind0010], T1['PGS'][ind0010], T1['GS'][ind0010], modelset=modelset))],
                [np.log10(T1['Temp'][ind0001]), T1['Logg'][ind0001], np.log10(T1['PGS'][ind0001]), T1['GS'][ind0001], 
                 np.log10(GetModel(T1['Temp'][ind0001], T1['Logg'][ind0001], T1['PGS'][ind0001], T1['GS'][ind0001], modelset=modelset))],
                [np.log10(T1['Temp'][ind1001]), T1['Logg'][ind1001], np.log10(T1['PGS'][ind1001]), T1['GS'][ind1001], 
                 np.log10(GetModel(T1['Temp'][ind1001], T1['Logg'][ind1001], T1['PGS'][ind1001], T1['GS'][ind1001], modelset=modelset))],
                [np.log10(T1['Temp'][ind0101]), T1['Logg'][ind0101], np.log10(T1['PGS'][ind0101]), T1['GS'][ind0101], 
                 np.log10(GetModel(T1['Temp'][ind0101], T1['Logg'][ind0101], T1['PGS'][ind0101], T1['GS'][ind0101], modelset=modelset))],
                [np.log10(T1['Temp'][ind0011]), T1['Logg'][ind0011], np.log10(T1['PGS'][ind0011]), T1['GS'][ind0011], 
                 np.log10(GetModel(T1['Temp'][ind0011], T1['Logg'][ind0011], T1['PGS'][ind0011], T1['GS'][ind0011], modelset=modelset))],
                [np.log10(T1['Temp'][ind1011]), T1['Logg'][ind1011], np.log10(T1['PGS'][ind1011]), T1['GS'][ind1011], 
                 np.log10(GetModel(T1['Temp'][ind1011], T1['Logg'][ind1011], T1['PGS'][ind1011], T1['GS'][ind1011], modelset=modelset))],
                [np.log10(T1['Temp'][ind0111]), T1['Logg'][ind0111], np.log10(T1['PGS'][ind0111]), T1['GS'][ind0111], 
                 np.log10(GetModel(T1['Temp'][ind0111], T1['Logg'][ind0111], T1['PGS'][ind0111], T1['GS'][ind0111], modelset=modelset))],
                [np.log10(T1['Temp'][ind1111]), T1['Logg'][ind1111], np.log10(T1['PGS'][ind1111]), T1['GS'][ind1111], 
                 np.log10(GetModel(T1['Temp'][ind1111], T1['Logg'][ind1111], T1['PGS'][ind1111], T1['GS'][ind1111], modelset=modelset))],
                [np.log10(T1['Temp'][ind0110]), T1['Logg'][ind0110], np.log10(T1['PGS'][ind0110]), T1['GS'][ind0110], 
                 np.log10(GetModel(T1['Temp'][ind0110], T1['Logg'][ind0110], T1['PGS'][ind0110], T1['GS'][ind0110], modelset=modelset))],
                [np.log10(T1['Temp'][ind1010]), T1['Logg'][ind1010], np.log10(T1['PGS'][ind1010]), T1['GS'][ind1010], 
                 np.log10(GetModel(T1['Temp'][ind1010], T1['Logg'][ind1010], T1['PGS'][ind1010], T1['GS'][ind1010], modelset=modelset))],
                [np.log10(T1['Temp'][ind1100]), T1['Logg'][ind1100], np.log10(T1['PGS'][ind1100]), T1['GS'][ind1100], 
                 np.log10(GetModel(T1['Temp'][ind1100], T1['Logg'][ind1100], T1['PGS'][ind1100], T1['GS'][ind1100], modelset=modelset))],
                [np.log10(T1['Temp'][ind1101]), T1['Logg'][ind1101], np.log10(T1['PGS'][ind1101]), T1['GS'][ind1101], 
                 np.log10(GetModel(T1['Temp'][ind1101], T1['Logg'][ind1101], T1['PGS'][ind1101], T1['GS'][ind1101], modelset=modelset))],
                [np.log10(T1['Temp'][ind1110]), T1['Logg'][ind1110], np.log10(T1['PGS'][ind1110]), T1['GS'][ind1110], 
                 np.log10(GetModel(T1['Temp'][ind1110], T1['Logg'][ind1110], T1['PGS'][ind1110], T1['GS'][ind1110], modelset=modelset))],
              ]
    #print(Points)
    waves2 = GetModel(T1['Temp'][ind1111], T1['Logg'][ind1111], T1['PGS'][ind1111], T1['GS'][ind1111], wave=True, modelset=modelset)
    #print('returning')
    #print(waves2)
    #print(ospf.utils.interpolations.quadlinear_interpolation(np.log10(Teff), Logg, np.log10(PGS), GS, Points))

    return waves2, ospf.utils.interpolations.quadlinear_interpolation(np.log10(Teff), Logg, np.log10(PGS), GS, Points)

