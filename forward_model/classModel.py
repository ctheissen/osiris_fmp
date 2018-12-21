#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import osiris_fmp as nsp
#import splat
#import splat.model as spmd

#def _constructModelName(teff, logg, feh, en, order, path=None):
#    """
#    Return the full name of the BT-Settl model.
#    """
#    if path is None:
#        path  = '/Users/dinohsu/projects/Models/models/btsettl08/' + \
#        'OSIRIS-' + str(band) + '-RAW/'
#    else:
#        path  = path + '/OSIRIS-' + str(band) + '-RAW/'
#    full_name = path + 'btsettl08_t'+ str(teff) + '_g' + \
#    '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + \
#    '_en' + '{0:.2f}'.format(float(en)) + '_OSIRIS-' + str(band) + '-RAW.txt'
#    
#    return full_name

class Model():
    """
    The Model class reads in the BT-SETTL models.    

    Parameters
    ----------
    1. Read in a BT-Settl model
    teff : int 
          The effective temperature, given from 500 to 3,500 K.
    logg : float
          The log(gravity), given in two decimal digits. 
          Ex: logg=4.50
    feh  : float
           The metalicity, given in two decimal digits. 
           Ex. feh=0.00
    en   : float
           alpha enhancement. given in two decimal digits. 
           Ex. en=0.00
    order: int
           The order of the model, given from 29 to 80
    path : str
           The path to the model

    2. Creat a model instance with given wavelengths and fluxes
    flux : astropy.table.column.Column
           The input flux.
    wave : astropy.table.column.Column
           The input wavelength.

    Returns
    -------
    flux : astropy.table.column.Column
           The flux retrieved from the model.
    wave : astropy.table.column.Column
           The wavelength retrieved from the model.

    Examples
    --------
    >>> import osiris_pip as nsp
    >>> model = nsp.Model(teff=2300, logg=5.5, order=33, path='/path/to/models')
    >>> model.plot()
    """
    def __init__(self, **kwargs):
        self.path  = kwargs.get('path')


        self.teff = kwargs.get('teff', 3000.)
        self.logg = kwargs.get('logg', 5.)
        self.feh  = kwargs.get('feh', 0)
        self.en   = kwargs.get('en', 0)
        self.modelset   = kwargs.get('modelset', 'aces2013')
        self.instrument = kwargs.get('instrument', 'OSIRIS')
        self.band       = kwargs.get('band', 'Kbb')
        if self.teff == None:
            self.teff = 2500
        if self.logg == None:
            self.logg = 5.00
        if self.feh  == None:
            self.feh  = 0.00
        if self.en   == None:
            self.en   = 0.00
        #print('Return a BT-Settl model of the order {0}, with Teff {1} logg {2}, z {3}, Alpha enhancement {4}.'\
        #    .format(self.order, self.teff, self.logg, self.feh, self.en))
    
        #full_name = _constructModelName(self.teff, self.logg, self.feh, self.en, self.order, self.path)
        #model = ascii.read(full_name, format='no_header', fast_reader=False)
        #self.wave  = model[0][:]*10000 #convert to Angstrom
        #self.flux  = model[1][:]
        
        ## load the splat.interpolation BTSETTL model
        #instrument = "OSIRIS-{}-RAW".format(self.band)
        #sp = spmd.getModel(instrument=str(instrument),teff=self.teff,logg=self.logg,z=self.feh)
        #self.wave = sp.wave.value*10000 #convert to Angstrom
        #self.flux = sp.flux.value

        #print('TEST1', self.order, self.instrument, self.band, self.modelset)
        
        wave, flux = nsp.forward_model.InterpolateModel.InterpModel(self.teff, self.logg, modelset=self.modelset, 
                                                                    instrument=self.instrument, band=self.band)
        self.wave = np.array(wave).flatten() * 10000 #convert to Angstrom
        self.flux = np.array(flux).flatten()


        

