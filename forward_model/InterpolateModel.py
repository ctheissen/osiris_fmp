import smart
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from numpy.linalg import inv, det
import osiris_fmp as ospf


##############################################################################################################


def InterpModel(teff, logg=4, metal=0, alpha=0, modelset='btsettl-cifist2011c', instrument='osiris', band='kbb'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set and instrument
    #if instrument.lower() == 'spex':
    if modelset.lower() == 'btsettl-cifist2011c':
        path     = BASE + '/../libraries/PHOENIX-BTSETTL-CIFIST2011C/%s-%s-RAW/'%(instrument.upper(), band.upper())
        Gridfile = BASE + '/../libraries/PHOENIX-BTSETTL-CIFIST2011C/PHOENIX-BTSETTL-CIFIST2011C_gridparams.csv'
    elif modelset.lower() == 'drift-phoenix':
        path     = BASE + '/../libraries/DRIFT-PHOENIX/%s-%s-RAW/'%(instrument.upper(), band.upper())
        Gridfile = BASE + '/../libraries/DRIFT-PHOENIX/DRIFT-PHOENIX_gridparams.csv'
    elif modelset.lower() == 'bt-dusty':
        path     = BASE + '/../libraries/BT-DUSTY/%s-%s-RAW/'%(instrument.upper(), band.upper())
        Gridfile = BASE + '/../libraries/BT-DUSTY/BT-DUSTY_gridparams.csv'
    elif modelset.lower() == 'phoenix-aces-agss-cond-2011':
        path     = BASE + '/../libraries/PHOENIX-ACES-AGSS-COND-2011/%s-%s-RAW/'%(instrument.upper(), band.upper())
        Gridfile = BASE + '/../libraries/PHOENIX-ACES-AGSS-COND-2011/PHOENIX_ACES_AGSS_COND_2011_gridparams.csv'
    elif modelset.lower() == 'sonora-2018': # Need to fix this
        path     = BASE + '/../libraries/SONORA-2018/%s-%s-RAW/'%(instrument.upper(), band)
        Gridfile = BASE + '/../libraries/SONORA-2018/SONORA_2018_gridparams.csv'
    '''
    elif instrument.lower() == 'osiris': # Still need to fix this one
        if modelset.lower() == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'
        elif modelset.lower() == 'drift-phoenix':
            path     = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011C/%s-%s-RAW/'%(instrument.upper(), band.upper())
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011C/PHOENIX_BTSETTL_CIFIST2011C_gridparams.csv'
        elif modelset.lower() == 'phoenix-btsettl-cifist2011-2015':
            path     = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/PHOENIX_BTSETTL_CIFIST2011_2015_gridparams_apogee.csv'
        elif modelset.lower() == 'phoenix-aces-agss-cond-2011' :
            path     = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams_apogee.csv'
        elif modelset.lower() == 'marcs-apogee-dr15' :
            path     = BASE + '/../libraries/MARCS_APOGEE_DR15/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/MARCS_APOGEE_DR15/MARCS_APOGEE_DR15_gridparams_apogee.csv'
    '''
    # Read the grid file
    #T1 = Table.read(Gridfile)
    T1 = np.genfromtxt(Gridfile, delimiter=',', names=True)
    #print(T1)
    #print(T1['Temp'])
    #sys.exit()

    ###################################################################################

    def GetModel(temp, wave=False, **kwargs):
        
        logg       = kwargs.get('logg', 4.5)
        metal      = kwargs.get('metal', 0)
        alpha      = kwargs.get('alpha', 0)
        kzz        = kwargs.get('kzz', 0)
        gridfile   = kwargs.get('gridfile', None)
        instrument = kwargs.get('instrument', 'nirspec')
        band       = kwargs.get('band', None)

        if gridfile is None:
            raise ValueError('Model gridfile must be provided.') 

        if modelset.lower() == 'drift-phoenix': 
                filename = 'DRIFT-PHOENIX_t'+ str(int(temp)) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_alpha' + '{0:.2f}'.format(float(alpha)) + '_kzz' + '{0:.2f}'.format(float(kzz)) + '_%s-%s-RAW.txt'%(instrument.upper(), band.upper()) 

        if modelset.lower() == 'bt-dusty': 
                filename = 'BT-DUSTY_t'+ str(int(temp)) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_alpha' + '{0:.2f}'.format(float(alpha)) + '_kzz' + '{0:.2f}'.format(float(kzz)) + '_%s-%s-RAW.txt'%(instrument.upper(), band.upper()) 

        if modelset.lower() == 'btsettl-cifist2011c': 
                filename = 'PHOENIX-BTSETTL-CIFIST2011C_t'+ str(int(temp)) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_alpha' + '{0:.2f}'.format(float(alpha)) + '_kzz' + '{0:.2f}'.format(float(kzz)) + '_%s-%s-RAW.txt'%(instrument.upper(), band.upper()) 

        if instrument.lower() == 'spex': 
            #print('SPEX')
            #print(temp, logg, alpha)
            
            if modelset == 'phoenix-aces-agss-cond-2011':
                filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp)) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_alpha{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            
            elif modelset == 'sonora-2018':
                filename = 'SONORA_2018_t{0:03d}'.format(int(temp)) + '_g{0:.2f}'.format(float(logg)) + '_FeH{0:.2f}'.format(0) + '_Y{0:.2f}'.format(0.28) + '_CO{0:.2f}'.format(1.00) + '_NIRSPEC-O' + str(order) + '-RAW.txt'

        #if instrument.lower() == 'osiris':
        #    filename = gridfile['File'][np.where( (gridfile['Temp']==temp) & (gridfile['Logg']==logg) & (gridfile['Metal']==metal) & (gridfile['Alpha']==alpha) )].data[0]

        #print(filename)
        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])
        #print(Tab['wave'])
        #print(Tab['flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    ###################################################################################

    # Check if the model already exists (grid point)
    if modelset == 'sonora-2018':
        if (teff, logg) in zip(T1['Temp'], T1['Logg']):
            metal, ys = 0, 0.28
            index0 = np.where( (T1['Temp'] == teff) & (T1['Logg'] == logg) & (T1['FeH'] == metal) & (T1['Y'] == ys) )
            #flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
            #waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, band=band, gridfile=T1)
            waves2 = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, band=band, gridfile=T1, wave=True)
            #print(waves2, flux2)
            return waves2, flux2
    else:
        if (teff, logg, metal, alpha) in zip(T1['Temp'], T1['Logg'], T1['Metal'], T1['Alpha']): 
            index0 = np.where( (T1['Temp'] == teff) & (T1['Logg'] == logg) & (T1['Metal'] == metal) & (T1['Alpha'] == alpha) )
            #print('INDEX', index0)
            #flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
            #waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['Metal'][index0], alpha=T1['Alpha'][index0], instrument=instrument, band=band, gridfile=T1)
            waves2 = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['Metal'][index0], alpha=T1['Alpha'][index0], instrument=instrument, band=band, gridfile=T1, wave=True)
            return waves2, flux2


    try:
        if modelset.lower() == 'sonora-2018':
            metal, alpha = 0, 0.28
            # Get the nearest models to the gridpoint (Temp)
            x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= teff)])
            x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= teff)])
            #print(x0, x1)
            
            # Get the nearest grid point to Logg
            y0 = np.max(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] <= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] <= logg) )])))
            y1 = np.min(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] >= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] >= logg) )])))
            #print(y0, y1)
            
            # Get the nearest grid point to [M/H]
            z0 = np.max(list(set(T1['FeH'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] <= metal) )]) & 
                             set(T1['FeH'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] <= metal) )])))
            z1 = np.min(list(set(T1['FeH'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] >= metal) )]) & 
                             set(T1['FeH'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] >= metal) )])))
            #print(z0, z1)
            
            # Get the nearest grid point to Alpha
            t0 = np.max(list(set(T1['Y'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] <= alpha) )]) & 
                             set(T1['Y'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] <= alpha) )])))
            t1 = np.min(list(set(T1['Y'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] >= alpha) )]) & 
                             set(T1['Y'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] >= alpha) )])))
            #print(t0, t1)
            
        else:
            # Get the nearest models to the gridpoint (Temp)
            x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= teff)])
            x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= teff)])
            #print('teff:', x0, teff, x1)
            # Get the nearest grid point to Logg
            y0 = np.max(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] <= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] <= logg) )])))
            y1 = np.min(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] >= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] >= logg) )])))
            #print('logg:', y0, logg, y1)
            # Get the nearest grid point to [M/H]
            #print(metal)
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) )])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) )])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] <= metal))])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= metal))])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] >= metal))])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= metal))])))
            z0 = np.max(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] <= metal) )]) & 
                             set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= metal) )])))
            z1 = np.min(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] >= metal) )]) & 
                             set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= metal) )])))
            #print('metal:', z0, metal, z1)
            # Get the nearest grid point to Alpha
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] <= alpha) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] <= alpha) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] >= alpha) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] >= alpha) )])))
            t0 = np.max(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] <= alpha) )]) & 
                             set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] <= alpha) )])))
            t1 = np.min(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] >= alpha) )]) & 
                             set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] >= alpha) )])))
            #print('alpha:', z0, alpha, z1)
    except:
        raise ValueError('Model Parameters Teff: %0.3f, Logg: %0.3f, [M/H]: %0.3f, Alpha: %0.3f are outside the model grid.'%(teff, logg, metal, alpha))


    if modelset.lower() == 'sonora-2018':
        # Get the 16 points
        ind0000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0000
        ind1000 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1000
        ind0100 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0100
        ind0010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0010
        ind0001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0001
        ind1001 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1001
        ind0101 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0101
        ind0011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0011
        ind1011 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1011
        ind0111 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0111
        ind1111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1111
        ind0110 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0110
        ind1010 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1010
        ind1100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1100
        ind1101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1101
        ind1110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1110
        Points =  [ [np.log10(T1['Temp'][ind0000]), T1['Logg'][ind0000], T1['FeH'][ind0000], T1['Y'][ind0000], 
                     np.log10(GetModel(T1['Temp'][ind0000], logg=T1['Logg'][ind0000], metal=T1['FeH'][ind0000], alpha=T1['Y'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1000]), T1['Logg'][ind1000], T1['FeH'][ind1000], T1['Y'][ind1000], 
                     np.log10(GetModel(T1['Temp'][ind1000], logg=T1['Logg'][ind1000], metal=T1['FeH'][ind1000], alpha=T1['Y'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0100]), T1['Logg'][ind0100], T1['FeH'][ind0100], T1['Y'][ind0100], 
                     np.log10(GetModel(T1['Temp'][ind0100], logg=T1['Logg'][ind0100], metal=T1['FeH'][ind0100], alpha=T1['Y'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0010]), T1['Logg'][ind0010], T1['FeH'][ind0010], T1['Y'][ind0010], 
                     np.log10(GetModel(T1['Temp'][ind0010], logg=T1['Logg'][ind0010], metal=T1['FeH'][ind0010], alpha=T1['Y'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0001]), T1['Logg'][ind0001], T1['FeH'][ind0001], T1['Y'][ind0001], 
                     np.log10(GetModel(T1['Temp'][ind0001], logg=T1['Logg'][ind0001], metal=T1['FeH'][ind0001], alpha=T1['Y'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1001]), T1['Logg'][ind1001], T1['FeH'][ind1001], T1['Y'][ind1001], 
                     np.log10(GetModel(T1['Temp'][ind1001], logg=T1['Logg'][ind1001], metal=T1['FeH'][ind1001], alpha=T1['Y'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0101]), T1['Logg'][ind0101], T1['FeH'][ind0101], T1['Y'][ind0101], 
                     np.log10(GetModel(T1['Temp'][ind0101], logg=T1['Logg'][ind0101], metal=T1['FeH'][ind0101], alpha=T1['Y'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0011]), T1['Logg'][ind0011], T1['FeH'][ind0011], T1['Y'][ind0011], 
                     np.log10(GetModel(T1['Temp'][ind0011], logg=T1['Logg'][ind0011], metal=T1['FeH'][ind0011], alpha=T1['Y'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1011]), T1['Logg'][ind1011], T1['FeH'][ind1011], T1['Y'][ind1011], 
                     np.log10(GetModel(T1['Temp'][ind1011], logg=T1['Logg'][ind1011], metal=T1['FeH'][ind1011], alpha=T1['Y'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0111]), T1['Logg'][ind0111], T1['FeH'][ind0111], T1['Y'][ind0111], 
                     np.log10(GetModel(T1['Temp'][ind0111], logg=T1['Logg'][ind0111], metal=T1['FeH'][ind0111], alpha=T1['Y'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1111]), T1['Logg'][ind1111], T1['FeH'][ind1111], T1['Y'][ind1111], 
                     np.log10(GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0110]), T1['Logg'][ind0110], T1['FeH'][ind0110], T1['Y'][ind0110], 
                     np.log10(GetModel(T1['Temp'][ind0110], logg=T1['Logg'][ind0110], metal=T1['FeH'][ind0110], alpha=T1['Y'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1010]), T1['Logg'][ind1010], T1['FeH'][ind1010], T1['Y'][ind1010], 
                     np.log10(GetModel(T1['Temp'][ind1010], logg=T1['Logg'][ind1010], metal=T1['FeH'][ind1010], alpha=T1['Y'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1100]), T1['Logg'][ind1100], T1['FeH'][ind1100], T1['Y'][ind1100], 
                     np.log10(GetModel(T1['Temp'][ind1100], logg=T1['Logg'][ind1100], metal=T1['FeH'][ind1100], alpha=T1['Y'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1101]), T1['Logg'][ind1101], T1['FeH'][ind1101], T1['Y'][ind1101], 
                     np.log10(GetModel(T1['Temp'][ind1101], logg=T1['Logg'][ind1101], metal=T1['FeH'][ind1101], alpha=T1['Y'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1110]), T1['Logg'][ind1110], T1['FeH'][ind1110], T1['Y'][ind1110], 
                     np.log10(GetModel(T1['Temp'][ind1110], logg=T1['Logg'][ind1110], metal=T1['FeH'][ind1110], alpha=T1['Y'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)
    else:
        # Get the 16 points
        #print(x0,x1,y0,y1,z0,z1,t0,t1)
        ind0000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 0000
        ind1000 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 1000
        ind0100 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 0100
        ind0010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 0010
        ind0001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 0001
        ind1001 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 1001
        ind0101 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 0101
        ind0011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 0011
        ind1011 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 1011
        ind0111 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 0111
        ind1111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 1111
        ind0110 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 0110
        ind1010 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 1010
        ind1100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 1100
        ind1101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 1101
        ind1110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 1110
        Points =  [ [np.log10(T1['Temp'][ind0000]), T1['Logg'][ind0000], T1['Metal'][ind0000], T1['Alpha'][ind0000], 
                     np.log10(GetModel(T1['Temp'][ind0000], logg=T1['Logg'][ind0000], metal=T1['Metal'][ind0000], alpha=T1['Alpha'][ind0000], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1000]), T1['Logg'][ind1000], T1['Metal'][ind1000], T1['Alpha'][ind1000], 
                     np.log10(GetModel(T1['Temp'][ind1000], logg=T1['Logg'][ind1000], metal=T1['Metal'][ind1000], alpha=T1['Alpha'][ind1000], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0100]), T1['Logg'][ind0100], T1['Metal'][ind0100], T1['Alpha'][ind0100], 
                     np.log10(GetModel(T1['Temp'][ind0100], logg=T1['Logg'][ind0100], metal=T1['Metal'][ind0100], alpha=T1['Alpha'][ind0100], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0010]), T1['Logg'][ind0010], T1['Metal'][ind0010], T1['Alpha'][ind0010], 
                     np.log10(GetModel(T1['Temp'][ind0010], logg=T1['Logg'][ind0010], metal=T1['Metal'][ind0010], alpha=T1['Alpha'][ind0010], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0001]), T1['Logg'][ind0001], T1['Metal'][ind0001], T1['Alpha'][ind0001], 
                     np.log10(GetModel(T1['Temp'][ind0001], logg=T1['Logg'][ind0001], metal=T1['Metal'][ind0001], alpha=T1['Alpha'][ind0001], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1001]), T1['Logg'][ind1001], T1['Metal'][ind1001], T1['Alpha'][ind1001], 
                     np.log10(GetModel(T1['Temp'][ind1001], logg=T1['Logg'][ind1001], metal=T1['Metal'][ind1001], alpha=T1['Alpha'][ind1001], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0101]), T1['Logg'][ind0101], T1['Metal'][ind0101], T1['Alpha'][ind0101], 
                     np.log10(GetModel(T1['Temp'][ind0101], logg=T1['Logg'][ind0101], metal=T1['Metal'][ind0101], alpha=T1['Alpha'][ind0101], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0011]), T1['Logg'][ind0011], T1['Metal'][ind0011], T1['Alpha'][ind0011], 
                     np.log10(GetModel(T1['Temp'][ind0011], logg=T1['Logg'][ind0011], metal=T1['Metal'][ind0011], alpha=T1['Alpha'][ind0011], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1011]), T1['Logg'][ind1011], T1['Metal'][ind1011], T1['Alpha'][ind1011], 
                     np.log10(GetModel(T1['Temp'][ind1011], logg=T1['Logg'][ind1011], metal=T1['Metal'][ind1011], alpha=T1['Alpha'][ind1011], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0111]), T1['Logg'][ind0111], T1['Metal'][ind0111], T1['Alpha'][ind0111], 
                     np.log10(GetModel(T1['Temp'][ind0111], logg=T1['Logg'][ind0111], metal=T1['Metal'][ind0111], alpha=T1['Alpha'][ind0111], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1111]), T1['Logg'][ind1111], T1['Metal'][ind1111], T1['Alpha'][ind1111], 
                     np.log10(GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['Metal'][ind1111], alpha=T1['Alpha'][ind1111], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0110]), T1['Logg'][ind0110], T1['Metal'][ind0110], T1['Alpha'][ind0110], 
                     np.log10(GetModel(T1['Temp'][ind0110], logg=T1['Logg'][ind0110], metal=T1['Metal'][ind0110], alpha=T1['Alpha'][ind0110], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1010]), T1['Logg'][ind1010], T1['Metal'][ind1010], T1['Alpha'][ind1010], 
                     np.log10(GetModel(T1['Temp'][ind1010], logg=T1['Logg'][ind1010], metal=T1['Metal'][ind1010], alpha=T1['Alpha'][ind1010], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1100]), T1['Logg'][ind1100], T1['Metal'][ind1100], T1['Alpha'][ind1100], 
                     np.log10(GetModel(T1['Temp'][ind1100], logg=T1['Logg'][ind1100], metal=T1['Metal'][ind1100], alpha=T1['Alpha'][ind1100], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1101]), T1['Logg'][ind1101], T1['Metal'][ind1101], T1['Alpha'][ind1101], 
                     np.log10(GetModel(T1['Temp'][ind1101], logg=T1['Logg'][ind1101], metal=T1['Metal'][ind1101], alpha=T1['Alpha'][ind1101], instrument=instrument, band=band, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1110]), T1['Logg'][ind1110], T1['Metal'][ind1110], T1['Alpha'][ind1110], 
                     np.log10(GetModel(T1['Temp'][ind1110], logg=T1['Logg'][ind1110], metal=T1['Metal'][ind1110], alpha=T1['Alpha'][ind1110], instrument=instrument, band=band, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['Metal'][ind1111], alpha=T1['Alpha'][ind1111], instrument=instrument, band=band, gridfile=T1, wave=True)

    return waves2, ospf.utils.interpolations.quadlinear_interpolation(np.log10(teff), logg, metal, alpha, Points)

