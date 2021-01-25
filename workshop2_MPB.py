import numpy as np
import pylab as plt
plt.ion()
import os


def binPhaseLC(phase, flux, nbins, cut_outliers=0):
    """
    Bins a lightcurve, typically phase-folded.

    Inputs
    -----------------
    phase: 			ndarray, N
        Phase data (could use a time array instead)
    flux:			ndarray, N
        Flux data
    nbins:			int
        Number of bins to use
    cut_outliers:	float
        If not zero, cuts outliers where (difference to median)/MAD > cut_outliers 
        	        
    Returns
    -----------------
    binnedlc:		ndarray, (nbins, 2)    
        Array of (bin phases, binned fluxes)
    """
    bin_edges = np.linspace(np.min(phase),np.max(phase),nbins)
    bin_indices = np.digitize(phase,bin_edges) - 1
    binnedlc = np.zeros([nbins,2])
    #fixes phase of all bins - means ignoring locations of points in bin
    binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  
    for bin in range(nbins):
        if np.sum(bin_indices==bin) > 0:
            inbin = np.where(bin_indices==bin)[0]
            if cut_outliers and np.sum(bin_indices==bin)>2:
                mad = np.median(np.abs(flux[inbin]-np.median(flux[inbin])))
                outliers = np.abs((flux[inbin] - np.median(flux[inbin])))/mad <= cut_outliers
                inbin = inbin[outliers]
            binnedlc[bin,1] = np.mean(flux[inbin])  
        else:
            #simple gap filling method, the main possible alternative is to interpolate.
            binnedlc[bin,1] = np.mean(flux)  
    return binnedlc


def bin_lightcurves(lcfile, bins=32, phasefold=False, offset=False, scale=False, frequencyfile=None, outfile=None):
    """
    Produces binned lightcurves and amplitudes (wrapper for binPhaseLC)

    Inputs
    -----------------
    lcfile			string
    				filepath to lightcurve data
    		
    bins			int
    				number of bins to use
    		
    phasefold		bool
    				if true, phase-fold lightcurve
    
    offset			bool
    				if true, offset lightcurve to minimum
    		
    scale			bool
    				if true, scale lightcurve so maximum is 1 and minimum is 0  	

    frequencyfile	string
    				filepath to frequency data 
    			
    outfile			string
    				filepath to save results to
    """
    if phasefold:
        assert frequencyfile is not None, 'If phasefold is True, file of frequencies must be provided.'
        assert os.path.exists(frequencyfile), 'frequencyfile not found'
    
    assert os.path.exists(lcfile), 'lcfile not found'
    assert bins > 0, 'bins must be a positive integer'
    assert isinstance(bins,int), 'bins must be a positive integer'

    dat = np.genfromtxt(lcfile)
    timearray = np.linspace(0,27.3820561,1341)
    output = np.ones([dat.shape[0],bins])
    amps = np.zeros(dat.shape[0])
        
    if frequencyfile is not None:
        freqs = np.genfromtxt(frequencyfile)
    else:
        freqs = np.zeros(dat.shape[0])
    
    for row in range(dat.shape[0]):
        flux = dat[row,:]
        if phasefold:
            period = 1./freqs[row,0]
            phase = np.mod(timearray,period)/period
        else:
            phase = timearray.copy()
            
        nancheck = np.isnan(flux)
        
        binnedlc = binPhaseLC(phase[~nancheck], flux[~nancheck], bins)
        
        if offset: #only valid for phasefolded lightcurves
            minimum = np.argmin(binnedlc[:,1])
            binnedlc[:,1] = np.roll(binnedlc[:,1],bins-minimum)
        
        amplitude = np.max(binnedlc[:,1]) - np.min(binnedlc[:,1])
        
        if scale:
            binnedlc[:,1] = (binnedlc[:,1]-np.min(binnedlc[:,1])) / amplitude
        
        output[row,:] = binnedlc[:,1]
        amps[row] = amplitude
    
    if outfile is not None:
        np.savetxt(outfile,output)
        np.savetxt(outfile[:-4]+'_amps.txt',amps)
        
    return dat, freqs, output, amps

     
     
def plotbinLC(dat, freqs, binlcs, Nrows=3, row=None):
    """
    Plots lightcurves from an array

    Inputs
    -----------------
    dat		ndarray, shape (n, 1341)
    		unbinned lightcurves
    		
    freqs	ndarray, shape(n, 3)
    		strongest three frequencies for each star
    		
    binlcs	ndarray, shape (n, nbins)
    		binned lightcurves
    
    Nrows	int
    		number of lightcurves to plot
    		
    row		list of int
    		list of row indices in dat of specific lightcurves to plot         	
    """
    import os
    import pylab as plt
    #plt.ion()
    
    
    timearray = np.linspace(0,27.3820561,1341)
    binx = np.arange(binlcs.shape[1])
    
    if row is not None:
        plt.clf()
        fig, axes = plt.subplots(len(row),3,figsize=(10,10),num=1)
        
        if len(row) == 1:
            axes[0].plot(timearray,dat[row[0],:],'.')
            
            period = 1./freqs[row[0],0]
            phase = np.mod(timearray,period)/period
            axes[1].plot(phase,dat[row[0],:],'.')
            axes[2].plot(binx,binlcs[row[0],:],'.')
            axes[1].set_title(str(row[0]))
            axes[0].set_ylabel('Rel. Flux')
            axes[0].set_xlabel('Time (days)')
            axes[1].set_xlabel('Phase (0-1)')
            axes[2].set_xlabel('Bins')
        else:
            for i in range(len(row)):
                axes[i,0].plot(timearray,dat[row[i],:],'.')
                period = 1./freqs[row[i],0]
                phase = np.mod(timearray,period)/period
                axes[i,1].plot(phase,dat[row[i],:],'.')
                axes[i,2].plot(binx,binlcs[row[i],:],'.')
                axes[i,1].set_title(str(row[i]))
                axes[i,0].set_ylabel('Rel. Flux')
            axes[len(row)-1,0].set_xlabel('Time (days)')
            axes[len(row)-1,1].set_xlabel('Phase (0-1)')
            axes[len(row)-1,2].set_xlabel('Bins')
    else:
        plt.clf()
        fig, axes = plt.subplots(Nrows,3,figsize=(10,10),num=1)
        rng = np.random.default_rng()
        rows = rng.choice(dat.shape[0],Nrows,replace=False)
        for i in range(Nrows):
            axes[i,0].plot(timearray,dat[rows[i],:],'.')
            period = 1./freqs[rows[i],0]
            phase = np.mod(timearray,period)/period
            axes[i,1].plot(phase,dat[rows[i],:],'.')
            axes[i,2].plot(binx,binlcs[rows[i],:],'.')
            axes[i,1].set_title(str(rows[i]))
            axes[i,0].set_ylabel('Rel. Flux')
        axes[Nrows-1,0].set_xlabel('Time (days)')
        axes[Nrows-1,1].set_xlabel('Phase (0-1)')
        axes[Nrows-1,2].set_xlabel('Bins')
    fig.tight_layout()
        
        
        
def p2p_features(flux):
    """
    Returns p2p features on lc

    Inputs
    -----------------
    flux	ndarray, shape(n)
    		flux data of a star
    
    Returns
    -----------------
    p2p 98th percentile: 	float
        98th percentile of point-to-point differences of lightcurve
    p2p mean:				float
        Mean of point-to-point differences of lightcurve
         	
    """
    p2p = np.abs(np.diff(flux))
    return np.percentile(p2p,98),np.mean(p2p)



def make_featureset(dat, freqs, amps, outfile): #could be full lcs or phased, binned version
    """
    Constructs non-lightcurve feature set. Saves to an '_features.txt' file

    Inputs
    -----------------
    dat				ndarray
    				lightcurve data

    freqs			ndarray
    				frequency data
    
    amps			ndarray
    				lightcurve amplitude data (output of bin_lightcurves)         	
    """
    from scipy.stats import skew,kurtosis
    
    ###change the second dimension (8) here to add new features
    features = np.zeros([dat.shape[0],8])
    ###
    
    for row in range(dat.shape[0]):
        flux = dat[row,:].copy()
        flux = flux[~np.isnan(flux)]
        features[row,0:3] = freqs[row]
        features[row,3] = amps[row]
        features[row,4:6] = p2p_features(flux)
        features[row,6] = skew(flux)
        features[row,7] = kurtosis(flux)
        ###
        #add new features here in same style
        ###
    
    np.savetxt(outfile,features)
