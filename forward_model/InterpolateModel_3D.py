import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from numpy.linalg import inv, det
import osiris_fmp as ospf

################################################################

def InterpModel_3D(Teff, Logg, PGS, modelset='aces-pso318', instrument='OSIRIS', band='Kbb'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the instrument and band
    if instrument.upper() == 'OSIRIS':
        bandname  = '%s-%s-RAW'%(instrument.upper(), band.upper())
    if instrument.upper() == 'CHARIS':
        bandname  = '%s-%s-RAW'%(instrument.upper(), band.upper())
    if instrument.upper() == 'SPEX':
        bandname  = '%s-%s-RAW'%(instrument.upper(), band.upper())

    # Check the model set
    if modelset.lower() == 'aces-pso318':
        path = BASE + '/../libraries/ACES-PSO318/%s/'%bandname.upper()
    elif modelset == 'agss09-dusty' :
        path = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-DUSTY/%s/'%bandname.upper()
    else:
        raise ValueError('Only aces-pso318 and agss09-dusty modelset available for 3D interpolation')
        

    def trilinear_interpolation(x, y, z, points):
        '''Interpolate (x,y) from values associated with 9 points.

        Custom routine

        '''

        (x0, y0, z0, q000), (x1, y0, z0, q100), (x0, y1, z0, q010), (x1, y1, z0, q110), \
        (x0, y0, z1, q001), (x1, y0, z1, q101), (x0, y1, z1, q011), (x1, y1, z1, q111),  = points
        x0 = x0.data[0]
        x1 = x1.data[0]
        y0 = y0.data[0]
        y1 = y1.data[0]
        z0 = z0.data[0]
        z1 = z1.data[0]
    
        #print(x0,x1,y0,y1,z0,z1)

        #print(x, y, x0.data, y0.data, _x0.data, y1.data, x1.data, _y0.data, _x1.data, _y1.data)
        #print(not x1 <= x <= x2, not y1 <= y <= y2)
        #if x0 != _x0 or x1 != _x1 or y0 != _y0 or y1 != _y1:
        #    raise ValueError('points do not form a rectangle')
        #if not x0 <= x <= x1 or not y0 <= y <= y1:
        #    raise ValueError('(x, y) not within the rectangle')

        c = np.array([ [1., x0, y0, z0, x0*y0, x0*z0, y0*z0, x0*y0*z0], #000
                       [1., x1, y0, z0, x1*y0, x1*z0, y0*z0, x1*y0*z0], #100
                       [1., x0, y1, z0, x0*y1, x0*z0, y1*z0, x0*y1*z0], #010
                       [1., x1, y1, z0, x1*y1, x1*z0, y1*z0, x1*y1*z0], #110
                       [1., x0, y0, z1, x0*y0, x0*z1, y0*z1, x0*y0*z1], #001
                       [1., x1, y0, z1, x1*y0, x1*z1, y0*z1, x1*y0*z1], #101
                       [1., x0, y1, z1, x0*y1, x0*z1, y1*z1, x0*y1*z1], #011
                       [1., x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1], #111
                      ], dtype='float')
        #print(c)
        #print(det(c))
        invc      = inv(c)
        transinvc = np.transpose(invc)

        final = np.dot(transinvc, [1, x, y, z, x*y, x*z, y*z, x*y*z])
        #print('Final Sum:', np.sum(final))


        interpFlux = 10**( (q000*final[0] + q100*final[1] + q010*final[2] + q110*final[3] + 
                            q001*final[4] + q101*final[5] + q011*final[6] + q111*final[7] ) )

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


    def GetModel(temp, logg, pgs, modelset='aces-pso318', wave=False):
        if modelset == 'aces-pso318':
            metal = 0
            alpha = 0
            gs    = 1
            kzz = int(1e8)
            filename = 'ACES-PSO318_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg.data[0])) + '_z' + '{0:.2f}'.format(metal) + '_alpha' + '{0:.2f}'.format(alpha) + '_pgs' + '{0:.2f}'.format(pgs.data[0]) +  '_gs' + '{0:.2f}'.format(gs) + '_kzz' + '{0:.2f}'.format(kzz) + '_%s.txt'%bandname
        elif modelset == 'agss09-dusty':
            kzz = 0.0
            feh = pgs
            filename = 'AGSS09-Dusty_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_Kzz{0:.1f}'.format(float(kzz)) + '_%s.txt'%bandname
        else:
            raise ValueError('Only aces-pso318 and agss09-dusty modelset available for 3D interpolation')
        #print(filename)
        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'], comment='#')
        #print('3', Tab['wave'].data)
        #print('3', Tab['flux'].data)

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']


    if modelset.lower() == 'aces-pso318':
        #Gridfile = BASE + '/../libraries/aces-pso318/aces-pso318_gridparams_uniform.csv'
        #Gridfile = BASE + '/../libraries/ACES-PSO318/ACES-PSO318_gridparams.csv'
        Gridfile = BASE + '/../libraries/ACES-PSO318/ACES-PSO318_gridparams_uniform.csv'
        T0 = Table.read(Gridfile, comment='#')
        T1 = T0[np.where( (T0['Kzz'] == 1e8) & (T0['GS'] == 1) ) ] # not using Kzz yet!

        # Check if the model already exists (grid point)
        if (Teff, Logg, PGS) in zip(T1['Temp'], T1['Logg'], T1['PGS']): 
            index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) & (T1['PGS'] == PGS) )
            flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['PGS'][index0], modelset=modelset )
            waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['PGS'][index0], modelset=modelset, wave=True)
            return waves2, flux2

        #x1     = np.floor(Teff/100.)*100
        #x2     = np.ceil(Teff/100.)*100
        #print('1', x1, x2)
        #y0, y1 = sorted(findlogg(Logg))

        # Get the nearest models to the gridpoint (Temp)
        #print(T1['Temp'][np.where(T1['Temp'] <= x1)])
        #print(T1['Temp'][np.where(T1['Temp'] >= x2)])
        x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= Teff)])
        x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= Teff)])
        print(x0, Teff, x1)
        #y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
        #y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        print(y0, Logg, y1)
        #z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
        #z1 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )][0]
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] <= PGS))])))
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] >= PGS))])))
        print(PGS)
        z0 = np.max(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] <= PGS)) )])))
        z1 = np.min(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] >= PGS)) )])))
        print(z0, PGS, z1)

        # Check if the gridpoint exists within the model ranges
        for x in [x0, x1]:
            for y in [y0, y1]:
                for z in [z0, z1]:
                    if (x, y, z) not in zip(T1['Temp'], T1['Logg'], T1['PGS']):
                        print('No Model', x, y, z)
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
        ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['PGS'] == z0) ) # 000
        ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['PGS'] == z0) ) # 100
        ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['PGS'] == z0) ) # 010
        ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] == z0) ) # 110
        ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['PGS'] == z1) ) # 001
        ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['PGS'] == z1) ) # 101
        ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['PGS'] == z1) ) # 011
        ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] == z1) ) # 111
        Points =  [ [np.log10(T1['Temp'][ind000]), T1['Logg'][ind000], np.log10(T1['PGS'][ind000]), 
                     np.log10(GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['PGS'][ind000], modelset=modelset))],
                    [np.log10(T1['Temp'][ind100]), T1['Logg'][ind100], np.log10(T1['PGS'][ind100]), 
                     np.log10(GetModel(T1['Temp'][ind100], T1['Logg'][ind100], T1['PGS'][ind100], modelset=modelset))],
                    [np.log10(T1['Temp'][ind010]), T1['Logg'][ind010], np.log10(T1['PGS'][ind010]),  
                     np.log10(GetModel(T1['Temp'][ind010], T1['Logg'][ind010], T1['PGS'][ind010], modelset=modelset))],
                    [np.log10(T1['Temp'][ind110]), T1['Logg'][ind110], np.log10(T1['PGS'][ind110]),  
                     np.log10(GetModel(T1['Temp'][ind110], T1['Logg'][ind110], T1['PGS'][ind110], modelset=modelset))],
                    [np.log10(T1['Temp'][ind001]), T1['Logg'][ind001], np.log10(T1['PGS'][ind001]), 
                     np.log10(GetModel(T1['Temp'][ind001], T1['Logg'][ind001], T1['PGS'][ind001], modelset=modelset))],
                    [np.log10(T1['Temp'][ind101]), T1['Logg'][ind101], np.log10(T1['PGS'][ind101]), 
                     np.log10(GetModel(T1['Temp'][ind101], T1['Logg'][ind101], T1['PGS'][ind101], modelset=modelset))],
                    [np.log10(T1['Temp'][ind011]), T1['Logg'][ind011], np.log10(T1['PGS'][ind011]), 
                     np.log10(GetModel(T1['Temp'][ind011], T1['Logg'][ind011], T1['PGS'][ind011], modelset=modelset))],
                    [np.log10(T1['Temp'][ind111]), T1['Logg'][ind111], np.log10(T1['PGS'][ind111]), 
                     np.log10(GetModel(T1['Temp'][ind111], T1['Logg'][ind111], T1['PGS'][ind111], modelset=modelset))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['PGS'][ind000], wave=True, modelset=modelset)

        #return waves2, trilinear_interpolation(np.log10(Teff), Logg, np.log10(PGS), Points)
        return waves2, ospf.utils.interpolations.trilinear_interpolation(np.log10(Teff), Logg, np.log10(PGS), Points)

    elif modelset == 'agss09-dusty':
        Gridfile = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/AGSS09-Dusty_gridparams.csv'

        T1 = Table.read(Gridfile)
        #T1 = T0[np.where( (T0['Kzz'] == 1e8) & (T0['gs'] == 1) ) ] # not using Kzz yet!

        # Check if the model already exists (grid point)
        #print(Teff, Logg, PGS)
        if (Teff, Logg, PGS) in zip(T1['Temp'], T1['Logg'], T1['Metal']): 
            #print('YES')
            index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) & (T1['Metal'] == PGS) )
            flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
            waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
            #print('3 waves', waves2)
            #print('3 flux', flux2)
            return waves2, flux2

        #x1     = np.floor(Teff/100.)*100
        #x2     = np.ceil(Teff/100.)*100
        #print('1', x1, x2)
        #y0, y1 = sorted(findlogg(Logg))

        # Get the nearest models to the gridpoint (Temp)
        #print(T1['Temp'][np.where(T1['Temp'] <= x1)])
        #print(T1['Temp'][np.where(T1['Temp'] >= x2)])
        x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= Teff)])
        x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= Teff)])
        #print(x0, 10**LogTeff, x1)
        #y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
        #y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
        #print(Logg)
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        #print(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        #print(y0, Logg, y1)
        #z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
        #z1 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )][0]
        #print(x0, y0, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] <= LogPGS))])))
        #print(x1, y1, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['Metal'] <= LogPGS))])))
        #print(x0, y0, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] >= LogPGS))])))
        #print(x1, y1, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['Metal'] >= LogPGS))])))
        z0 = np.max(list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] <= PGS) )]) & set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= PGS)) )])))
        z1 = np.min(list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] >= PGS) )]) & set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= PGS)) )])))
        #print('PGS:', z0, LogPGS, z1)

        # Check if the gridpoint exists within the model ranges
        for x in [x0, x1]:
            for y in [y0, y1]:
                for z in [z0, z1]:
                    if (x, y, z) not in zip(T1['Temp'], T1['Logg'], T1['Metal']):
                        print('No Model', x, y, z)
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
        ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) ) # 000
        ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) ) # 100
        ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) ) # 010
        ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) ) # 110
        ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) ) # 001
        ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) ) # 101
        ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) ) # 011
        ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) ) # 111
        Points =  [ [np.log10(T1['Temp'][ind000]), T1['Logg'][ind000], T1['Metal'][ind000], 
                     np.log10(GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['Metal'][ind000], modelset=modelset))],
                    [np.log10(T1['Temp'][ind100]), T1['Logg'][ind100], T1['Metal'][ind100], 
                     np.log10(GetModel(T1['Temp'][ind100], T1['Logg'][ind100], T1['Metal'][ind100], modelset=modelset))],
                    [np.log10(T1['Temp'][ind010]), T1['Logg'][ind010], T1['Metal'][ind010],  
                     np.log10(GetModel(T1['Temp'][ind010], T1['Logg'][ind010], T1['Metal'][ind010], modelset=modelset))],
                    [np.log10(T1['Temp'][ind110]), T1['Logg'][ind110], T1['Metal'][ind110],  
                     np.log10(GetModel(T1['Temp'][ind110], T1['Logg'][ind110], T1['Metal'][ind110], modelset=modelset))],
                    [np.log10(T1['Temp'][ind001]), T1['Logg'][ind001], T1['Metal'][ind001], 
                     np.log10(GetModel(T1['Temp'][ind001], T1['Logg'][ind001], T1['Metal'][ind001], modelset=modelset))],
                    [np.log10(T1['Temp'][ind101]), T1['Logg'][ind101], T1['Metal'][ind101], 
                     np.log10(GetModel(T1['Temp'][ind101], T1['Logg'][ind101], T1['Metal'][ind101], modelset=modelset))],
                    [np.log10(T1['Temp'][ind011]), T1['Logg'][ind011], T1['Metal'][ind011], 
                     np.log10(GetModel(T1['Temp'][ind011], T1['Logg'][ind011], T1['Metal'][ind011], modelset=modelset))],
                    [np.log10(T1['Temp'][ind111]), T1['Logg'][ind111], T1['Metal'][ind111], 
                     np.log10(GetModel(T1['Temp'][ind111], T1['Logg'][ind111], T1['Metal'][ind111], modelset=modelset))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['Metal'][ind000], wave=True, modelset=modelset)

        return waves2, trilinear_interpolation(np.log10(Teff), Logg, PGS, Points)



####################################################################################################################################





def InterpModel_Log3D(LogTeff, Logg, LogPGS, modelset='aces-pso318', instrument='OSIRIS', band='Kbb'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    #print('PARAMS:', LogTeff, Logg, LogPGS)

    # Check the instrument and band
    if instrument == 'OSIRIS':
        bandname  = '%s-%s-RAW'%(instrument, band)
    if instrument == 'CHARIS':
        bandname  = '%s-%s-RAW'%(instrument, band)

    # Check the model set
    if modelset == 'aces-pso318':
        path = BASE + '/../libraries/aces-pso318/%s/'%bandname
    elif modelset == 'agss09-dusty' :
        path = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/%s/'%bandname
    else:
        raise ValueError('Only aces-pso318 and agss09-dusty modelset available for 3D interpolation')
        

    def trilinear_interpolation(x, y, z, points):
        '''Interpolate (x,y) from values associated with 9 points.

        Custom routine

        '''

        (x0, y0, z0, q000), (x1, y0, z0, q100), (x0, y1, z0, q010), (x1, y1, z0, q110), \
        (x0, y0, z1, q001), (x1, y0, z1, q101), (x0, y1, z1, q011), (x1, y1, z1, q111),  = points
        x0 = x0.data[0]
        x1 = x1.data[0]
        y0 = y0.data[0]
        y1 = y1.data[0]
        z0 = z0.data[0]
        z1 = z1.data[0]
    
        #print('XYZ:', x0,x1,y0,y1,z0,z1)

        #print(x, y, x0.data, y0.data, _x0.data, y1.data, x1.data, _y0.data, _x1.data, _y1.data)
        #print(not x1 <= x <= x2, not y1 <= y <= y2)
        #if x0 != _x0 or x1 != _x1 or y0 != _y0 or y1 != _y1:
        #    raise ValueError('points do not form a rectangle')
        #if not x0 <= x <= x1 or not y0 <= y <= y1:
        #    raise ValueError('(x, y) not within the rectangle')

        c = np.array([ [1., x0, y0, z0, x0*y0, x0*z0, y0*z0, x0*y0*z0], #000
                       [1., x1, y0, z0, x1*y0, x1*z0, y0*z0, x1*y0*z0], #100
                       [1., x0, y1, z0, x0*y1, x0*z0, y1*z0, x0*y1*z0], #010
                       [1., x1, y1, z0, x1*y1, x1*z0, y1*z0, x1*y1*z0], #110
                       [1., x0, y0, z1, x0*y0, x0*z1, y0*z1, x0*y0*z1], #001
                       [1., x1, y0, z1, x1*y0, x1*z1, y0*z1, x1*y0*z1], #101
                       [1., x0, y1, z1, x0*y1, x0*z1, y1*z1, x0*y1*z1], #011
                       [1., x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1], #111
                      ], dtype='float')
        #print(c)
        #print(det(c))
        invc      = inv(c)
        transinvc = np.transpose(invc)

        final = np.dot(transinvc, [1, x, y, z, x*y, x*z, y*z, x*y*z])
        #print('Final Sum:', np.sum(final))


        interpFlux = 10**( (q000*final[0] + q100*final[1] + q010*final[2] + q110*final[3] + 
                            q001*final[4] + q101*final[5] + q011*final[6] + q111*final[7] ) )
        #print('Interp Flux:', interpFlux)

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


    def GetModel(temp, logg, pgs, modelset='aces-pso318', wave=False):
  
        if modelset == 'aces-pso318':
            filename = 'aces-pso318_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg.data[0])) + '_pgs' + '{}'.format(pgs.data[0]) + '_Kzz' + '{}'.format(kzz) + '_gs' + '{0:.1f}'.format(gs) + '_%s.txt'%bandname
            kzz = int(1e8)
            gs  = 1.
        elif modelset == 'agss09-dusty':
            kzz = 0.0
            feh = pgs
            #print('{0:03d}'.format(int(temp.data[0])))
            #print('_g{0:.2f}'.format(float(logg)))
            #print('_z{0:.2f}'.format(float(feh)))
            #print('_Kzz{0:.1f}'.format(float(kzz)))
            filename = 'AGSS09-Dusty_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_Kzz{0:.1f}'.format(float(kzz)) + '_%s.txt'%bandname
        else:
            raise ValueError('Only aces-pso318 and agss09-dusty modelset available for 3D interpolation')
        
        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])
        #print('3', Tab['wave'].data)
        #print('3', Tab['flux'].data)

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']
    '''
    def myround(x, base=.5):
        return base * round(float(x)/base)

    def findlogg(logg):
        LoggArr = np.arange(2.5, 6, 0.5)
        dist    = (LoggArr - logg)**2
        return LoggArr[np.argsort(dist)][0:2]
    '''
    if modelset == 'aces-pso318':
        Gridfile = BASE + '/../libraries/aces-pso318/aces-pso318_gridparams_uniform.csv'

        T0 = Table.read(Gridfile)
        T1 = T0[np.where( (T0['Kzz'] == 1e8) & (T0['gs'] == 1) ) ] # not using Kzz yet!

        # Check if the model already exists (grid point)
        if (10**LogTeff, Logg, LogPGS) in zip(T1['Temp'], T1['Logg'], T1['pgs']): 
            index0 = np.where( (T1['Temp'] == 10**LogTeff) & (T1['Logg'] == Logg) & (T1['pgs'] == 10**LogPGS) )
            flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['pgs'][index0], modelset=modelset )
            waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['pgs'][index0], modelset=modelset, wave=True)
            return waves2, flux2

        #x1     = np.floor(Teff/100.)*100
        #x2     = np.ceil(Teff/100.)*100
        #print('1', x1, x2)
        #y0, y1 = sorted(findlogg(Logg))

        # Get the nearest models to the gridpoint (Temp)
        #print(T1['Temp'][np.where(T1['Temp'] <= x1)])
        #print(T1['Temp'][np.where(T1['Temp'] >= x2)])
        x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= 10**LogTeff)])
        x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= 10**LogTeff)])
        #print(x0, Teff, x1)
        #y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
        #y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        #print(y0, Logg, y1)
        #z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
        #z1 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )][0]
        #print(x0, y0, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['pgs'] <= PGS))])))
        #print(x1, y1, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS))])))
        #print(x0, y0, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['pgs'] >= PGS))])))
        #print(x1, y1, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS))])))
        z0 = np.max(list(set(T1['pgs'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['pgs'] <= 10**LogPGS) )]) & set(T1['pgs'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] <= 10**LogPGS)) )])))
        z1 = np.min(list(set(T1['pgs'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['pgs'] >= 10**LogPGS) )]) & set(T1['pgs'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] >= 10**LogPGS)) )])))
        #print('PGS:', z0, PGS, z1)

        # Check if the gridpoint exists within the model ranges
        for x in [x0, x1]:
            for y in [y0, y1]:
                for z in [z0, z1]:
                    if (x, y, z) not in zip(T1['Temp'], T1['Logg'], T1['pgs']):
                        print('No Model', x, y, z)
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
        ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['pgs'] == z0) ) # 000
        ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['pgs'] == z0) ) # 100
        ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['pgs'] == z0) ) # 010
        ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] == z0) ) # 110
        ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['pgs'] == z1) ) # 001
        ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['pgs'] == z1) ) # 101
        ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['pgs'] == z1) ) # 011
        ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['pgs'] == z1) ) # 111
        Points =  [ [np.log10(T1['Temp'][ind000]), T1['Logg'][ind000], np.log10(T1['pgs'][ind000]), 
                     np.log10(GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['pgs'][ind000], modelset=modelset))],
                    [np.log10(T1['Temp'][ind100]), T1['Logg'][ind100], np.log10(T1['pgs'][ind100]), 
                     np.log10(GetModel(T1['Temp'][ind100], T1['Logg'][ind100], T1['pgs'][ind100], modelset=modelset))],
                    [np.log10(T1['Temp'][ind010]), T1['Logg'][ind010], np.log10(T1['pgs'][ind010]),  
                     np.log10(GetModel(T1['Temp'][ind010], T1['Logg'][ind010], T1['pgs'][ind010], modelset=modelset))],
                    [np.log10(T1['Temp'][ind110]), T1['Logg'][ind110], np.log10(T1['pgs'][ind110]),  
                     np.log10(GetModel(T1['Temp'][ind110], T1['Logg'][ind110], T1['pgs'][ind110], modelset=modelset))],
                    [np.log10(T1['Temp'][ind001]), T1['Logg'][ind001], np.log10(T1['pgs'][ind001]), 
                     np.log10(GetModel(T1['Temp'][ind001], T1['Logg'][ind001], T1['pgs'][ind001], modelset=modelset))],
                    [np.log10(T1['Temp'][ind101]), T1['Logg'][ind101], np.log10(T1['pgs'][ind101]), 
                     np.log10(GetModel(T1['Temp'][ind101], T1['Logg'][ind101], T1['pgs'][ind101], modelset=modelset))],
                    [np.log10(T1['Temp'][ind011]), T1['Logg'][ind011], np.log10(T1['pgs'][ind011]), 
                     np.log10(GetModel(T1['Temp'][ind011], T1['Logg'][ind011], T1['pgs'][ind011], modelset=modelset))],
                    [np.log10(T1['Temp'][ind111]), T1['Logg'][ind111], np.log10(T1['pgs'][ind111]), 
                     np.log10(GetModel(T1['Temp'][ind111], T1['Logg'][ind111], T1['pgs'][ind111], modelset=modelset))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['pgs'][ind000], wave=True, modelset=modelset)

        return waves2, trilinear_interpolation(LogTeff, Logg, LogPGS, Points)


    elif modelset == 'agss09-dusty':
        Gridfile = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-Dusty/AGSS09-Dusty_gridparams.csv'

        T1 = Table.read(Gridfile)
        #T1 = T0[np.where( (T0['Kzz'] == 1e8) & (T0['gs'] == 1) ) ] # not using Kzz yet!

        # Check if the model already exists (grid point)
        #print(10**LogTeff, Logg, LogPGS)
        if (10**LogTeff, Logg, LogPGS) in zip(T1['Temp'], T1['Logg'], T1['Metal']): 
            index0 = np.where( (T1['Temp'] == 10**LogTeff) & (T1['Logg'] == Logg) & (T1['Metal'] == LogPGS) )
            flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
            waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
            #print('3 waves', waves2)
            #print('3 flux', flux2)
            return waves2, flux2

        #x1     = np.floor(Teff/100.)*100
        #x2     = np.ceil(Teff/100.)*100
        #print('1', x1, x2)
        #y0, y1 = sorted(findlogg(Logg))

        # Get the nearest models to the gridpoint (Temp)
        #print(T1['Temp'][np.where(T1['Temp'] <= x1)])
        #print(T1['Temp'][np.where(T1['Temp'] >= x2)])
        x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= 10**LogTeff)])
        x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= 10**LogTeff)])
        #print(x0, 10**LogTeff, x1)
        #y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
        #y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
        #print(Logg)
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        #print(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        #print(y0, Logg, y1)
        #z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
        #z1 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )][0]
        #print(x0, y0, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] <= LogPGS))])))
        #print(x1, y1, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['Metal'] <= LogPGS))])))
        #print(x0, y0, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] >= LogPGS))])))
        #print(x1, y1, list(set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['Metal'] >= LogPGS))])))
        z0 = np.max(list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] <= LogPGS) )]) & set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= LogPGS)) )])))
        z1 = np.min(list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] >= LogPGS) )]) & set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= LogPGS)) )])))
        #print('PGS:', z0, LogPGS, z1)

        # Check if the gridpoint exists within the model ranges
        for x in [x0, x1]:
            for y in [y0, y1]:
                for z in [z0, z1]:
                    if (x, y, z) not in zip(T1['Temp'], T1['Logg'], T1['Metal']):
                        print('No Model', x, y, z)
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
        ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) ) # 000
        ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) ) # 100
        ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) ) # 010
        ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) ) # 110
        ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) ) # 001
        ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) ) # 101
        ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) ) # 011
        ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) ) # 111
        Points =  [ [np.log10(T1['Temp'][ind000]), T1['Logg'][ind000], T1['Metal'][ind000], 
                     np.log10(GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['Metal'][ind000], modelset=modelset))],
                    [np.log10(T1['Temp'][ind100]), T1['Logg'][ind100], T1['Metal'][ind100], 
                     np.log10(GetModel(T1['Temp'][ind100], T1['Logg'][ind100], T1['Metal'][ind100], modelset=modelset))],
                    [np.log10(T1['Temp'][ind010]), T1['Logg'][ind010], T1['Metal'][ind010],  
                     np.log10(GetModel(T1['Temp'][ind010], T1['Logg'][ind010], T1['Metal'][ind010], modelset=modelset))],
                    [np.log10(T1['Temp'][ind110]), T1['Logg'][ind110], T1['Metal'][ind110],  
                     np.log10(GetModel(T1['Temp'][ind110], T1['Logg'][ind110], T1['Metal'][ind110], modelset=modelset))],
                    [np.log10(T1['Temp'][ind001]), T1['Logg'][ind001], T1['Metal'][ind001], 
                     np.log10(GetModel(T1['Temp'][ind001], T1['Logg'][ind001], T1['Metal'][ind001], modelset=modelset))],
                    [np.log10(T1['Temp'][ind101]), T1['Logg'][ind101], T1['Metal'][ind101], 
                     np.log10(GetModel(T1['Temp'][ind101], T1['Logg'][ind101], T1['Metal'][ind101], modelset=modelset))],
                    [np.log10(T1['Temp'][ind011]), T1['Logg'][ind011], T1['Metal'][ind011], 
                     np.log10(GetModel(T1['Temp'][ind011], T1['Logg'][ind011], T1['Metal'][ind011], modelset=modelset))],
                    [np.log10(T1['Temp'][ind111]), T1['Logg'][ind111], T1['Metal'][ind111], 
                     np.log10(GetModel(T1['Temp'][ind111], T1['Logg'][ind111], T1['Metal'][ind111], modelset=modelset))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['Metal'][ind000], wave=True, modelset=modelset)

        return waves2, trilinear_interpolation(LogTeff, Logg, LogPGS, Points)

