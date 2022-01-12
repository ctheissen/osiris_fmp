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
    bandname  = '%s-%s-RAW'%(instrument.upper(), band.upper())

    # Check the model set
    if modelset.lower() == 'aces-pso318':
        path = BASE + '/../libraries/ACES-PSO318/%s/'%bandname.upper()
    elif modelset == 'agss09-dusty' :
        path = BASE + '/../libraries/PHOENIX-ACES/2019/AGSS09-DUSTY/%s/'%bandname.upper()
    elif modelset == 'hr8799c' :
        path = BASE + '/../libraries/HR8799C/%s/'%bandname.upper()
    if modelset.lower() == 'vhs1256-pso':
        path = BASE + '/../libraries/VHS1256_PSO/%s/'%bandname
    else:
        raise ValueError('Only [aces-pso318, agss09-dusty, hr8799c, vhs1256-pso] modelsets available for 3D interpolation')


    def GetModel(temp, logg, pgs, modelset='aces-pso318', wave=False):

        if modelset.lower() == 'aces-pso318':
            metal = 0
            alpha = 0
            gs    = 1
            kzz = int(1e8)
            filename = 'ACES-PSO318_t'+ str(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg.data[0])) + '_z{0:.2f}'.format(metal) + '_alpha{0:.2f}'.format(alpha) + '_pgs{0:.2f}'.format(pgs.data[0]) +  '_gs{0:.2f}'.format(gs) + '_kzz{0:.2f}'.format(kzz) + '_%s.txt'%bandname
        elif modelset.lower() == 'hr8799c':
            metal = 0
            alpha = 0
            gs    = 1
            kzz = int(1e8)
            filename = 'HR8799C_t'+ str(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg.data[0])) + '_z{0:.2f}'.format(metal) + '_alpha{0:.2f}'.format(alpha) + '_pgs{0:.2f}'.format(pgs.data[0]) +  '_gs{0:.2f}'.format(gs) + '_kzz{0:.2f}'.format(kzz) + '_%s.txt'%bandname
        elif modelset.lower() == 'agss09-dusty':
            kzz = 0.0
            feh = pgs
            filename = 'AGSS09-Dusty_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_Kzz{0:.1f}'.format(float(kzz)) + '_%s.txt'%bandname    
        elif modelset.lower() == 'vhs1256-pso':
            metal = 0
            alpha = 0
            kzz   = int(1e8)
            gs    = 1
            filename = 'VHS1256_PSO_t'+ str(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg.data[0])) + '_z{0:.2f}'.format(metal) + '_alpha{0:.2f}'.format(alpha) + '_pgs{0:.2f}'.format(pgs.data[0]) + '_gs{0:.2f}'.format(gs) + '_kzz{0:.2f}'.format(kzz) + '_%s.txt'%bandname
            #print('File', filename)
        else:
            raise ValueError('Only [aces-pso318, agss09-dusty, hr8799c, vhs1256-pso] modelsets available for 3D interpolation')
        #print(filename)
        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'], comment='#')
        #print('3', Tab['wave'].data)
        #print('3', Tab['flux'].data)

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']


    if modelset.lower() == 'hr8799c':
        Gridfile = BASE + '/../libraries/HR8799C/HR8799C_gridparams.csv'
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
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] <= PGS))])))
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] >= PGS))])))
        #print(PGS)
        z0 = np.max(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] <= PGS)) )])))
        z1 = np.min(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] >= PGS)) )])))
        #print(z0, PGS, z1)

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
        waves2 = GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['PGS'][ind000], wave=True, modelset=modelset)

        return waves2, ospf.utils.interpolations.trilinear_interpolation(np.log10(Teff), Logg, np.log10(PGS), Points)


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
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] <= PGS))])))
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] >= PGS))])))
        #print(PGS)
        z0 = np.max(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] <= PGS)) )])))
        z1 = np.min(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] >= PGS)) )])))
        #print(z0, PGS, z1)

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


    elif modelset.lower() == 'vhs1256-pso':
        #Gridfile = BASE + '/../libraries/aces-pso318/aces-pso318_gridparams_uniform.csv'
        #Gridfile = BASE + '/../libraries/ACES-PSO318/ACES-PSO318_gridparams.csv'
        Gridfile = BASE + '/../libraries/VHS1256_PSO/VHS1256_PSO_gridparams.csv'
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
        #print(x0, Teff, x1)
        y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
        y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )])))
        #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
        y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
        #print(y0, Logg, y1)
        #z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
        #z1 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )][0]
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] <= PGS))])))
        #print(x0, y0, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS))])))
        #print(x1, y1, list(set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['PGS'] >= PGS))])))
        #print(PGS)
        z0 = np.max(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] <= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] <= PGS)) )])))
        z1 = np.min(list(set(T1['PGS'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['PGS'] >= PGS) )]) & set(T1['PGS'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['PGS'] >= PGS)) )])))
        #print(z0, PGS, z1)

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

        return waves2, ospf.utils.trilinear_interpolation(np.log10(Teff), Logg, PGS, Points)



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

        return waves2, ospf.utils.interpolations.trilinear_interpolation(LogTeff, Logg, LogPGS, Points)


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

        return waves2, ospf.utils.interpolations.trilinear_interpolation(LogTeff, Logg, LogPGS, Points)

