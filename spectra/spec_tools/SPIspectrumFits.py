import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spec_tools.tsiegert_tools import *
from spec_tools.spectral_shapes import *
from spec_tools.priors import *
import emcee
import corner
import time


def SPI_model_fit(theta, x, y, dx, y_err, rsp, e_lo, e_hi, function, eval=False):
    """
    Returns:
    Negative normal-distributed log-likelihood for SPI data fitted with 
    arbitraty spectral model, accounting for the spectral response (default).
    Or, with eval=True, the output model in data space (in units of cnts/cm2/s/keV)
    
    Parameters:
    :param theta:     Array of to-be-fitted parameters used in combination with defined 'function'
    :param x:         Energy array (dim=(n,); in keV)
    :param y:         Differential flux array (dim=(n,); in cnts/cm2/s/keV)
    :param dx:        Energy bin sizes for x (dim=(n,); in keV)
    :param y_err:     Uncertainties on differential fluxes (one sigma values) (dim=(n,); in cnts/cm2/s/keV)
    :param rsp:       Response matrix that fits to the dimensions of x (and dx, y, y_err; dim=(m,n))
    :param e_lo:      Lower interpolated energy edges for response calculation(dim=(m,); in keV)
    :param e_hi:      Upper interpolated energy edges for response calculation(dim=(m,); in keV)  
    :param function:  String of named function in the current(!) notebook (may change later if individual functions are written in files or something)
    """
    
    # Now, same things as above:
    # Integrate model with Simpson's rule over the interpolated energy bins
    integrated_model = (e_hi-e_lo)/6.0*(globals()[function](e_lo,theta)+
                                        4*globals()[function]((e_lo+e_hi)/2.0,theta)+
                                        globals()[function](e_hi,theta))
    
    # Apply response matrix
    folded_model = np.dot(integrated_model,rsp)
 
    # Return to differential model
    folded_differential_model = folded_model / dx

    # Evaluate either chi2
    if eval==False:
        return -0.5*np.nansum((y-folded_differential_model)**2/y_err**2)
    # or return the folded model itself at a certain set of parameters
    else:
        return folded_differential_model
    
    
# add prior for each parameter to be fitted according to definition in 'function'
def ln_prior_SPI_spectrum(theta,prior):
    """
    this function assumes prior dictionaries of the shape:
    prior = {0: ('normal_prior',1e-4,1e-5,'Amplitude'),
             1: ('uniform_prior',-1.8,-1.6,'Index')}
    with the key being numbered sequentially
    then there is a tag for the prior function (normal, uniform, whatever you define else)
    then the prior values as defined in the functions
    and a name to identifiy later
    NO CHECKS FOR CONSISTENCY ARE PERFORMED
    """
    lnprior = 0.
    for p in range(len(theta)):
        lnprior += globals()[prior[p][0]](theta[p],prior[p][1:])
    return lnprior


def ln_posterior_SPI_spectrum(theta, x, y, dx, y_err, rsp, e_lo, e_hi, function, prior=None):
    if prior != None:
        lp = ln_prior_SPI_spectrum(theta,prior)
    else:
        lp = 0.
    if not np.isfinite(lp):
        return -np.inf
    return lp + SPI_model_fit(theta, x, y, dx, y_err, rsp, e_lo, e_hi, function)


def define_rsp_matrix(rsp_file):
    #rsp_file = 'spectral_response.rmf.fits'
    hdul = fits.open(rsp_file)
    hdul.info()
    hdul['EBOUNDS'].data.columns
    e_min = hdul['EBOUNDS'].data['E_MIN']
    e_max = hdul['EBOUNDS'].data['E_MAX']
    ee = (e_max+e_min)/2. # centre of the bin
    dee = (e_max-e_min)   # bin size
    hdul['MATRIX'].data.columns
    e_lo = hdul['MATRIX'].data['ENERG_LO']
    e_hi = hdul['MATRIX'].data['ENERG_HI']
    e_mean = (e_hi+e_lo)/2.
    de = (e_hi-e_lo)

    matrix = np.zeros((len(e_lo),len(e_min)))
    for i in range(len(e_lo)):
        matrix[i,0:hdul['MATRIX'].data['N_CHAN'][i]] = hdul['MATRIX'].data['MATRIX'][i]

    hdul.close()

    return matrix,e_lo,e_hi,e_min,e_max




def fit_SPI_spec(ee,flux,dee,flux_err,matrix,e_lo,e_hi,fit_function,guess,prior=None,iters=2000):

    # number of parameters
    n_par = len(guess)

    # emcee workflow
    ndim, nwalkers = n_par, n_par*5
    pos = [guess + np.random.randn(ndim)*1e-4 for i in range(nwalkers)]

    # set time
    start = time.time()

    # set up sampler
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    ln_posterior_SPI_spectrum,
                                    args=(ee,
                                          flux,
                                          dee,
                                          flux_err,
                                          matrix,
                                          e_lo,
                                          e_hi,
                                          fit_function,
                                          prior))

    # run smapler
    _ = sampler.run_mcmc(pos, iters, progress=True)

    # check how long the fit took
    end = time.time()

    # extract samples
    samples = sampler.get_chain()
    samplesf = sampler.flatchain

    n_samples = iters#*nwalkers
    n_walkers = nwalkers

    burnin = int(0.5*n_samples)

    ttime = end - start
    print("Processing took {0:.1f} seconds".format(ttime))

    # print results
    print('\n')
    print('Results:\n')

    spec_params = np.zeros((n_par,7))


    # formatting the table
    row_format ='{:>10}' * 8

    # first table row
    print(row_format.format(*['Parameter','mean','std','0.15','15.85','50.00','84.15','99.85']))

    for p in range(n_par):
        mean_val   = np.mean(samples[burnin:,:,p])
        std_val    = np.std(samples[burnin:,:,p])
        median_val = np.median(samples[burnin:,:,p])
        ub1_val    = np.percentile(samples[burnin:,:,p],50+68.3/2)
        lb1_val    = np.percentile(samples[burnin:,:,p],50-68.3/2)
        ub3_val    = np.percentile(samples[burnin:,:,p],50+99.73/2)
        lb3_val    = np.percentile(samples[burnin:,:,p],50-99.73/2)
        spec_params[p,:] = [mean_val,std_val,lb3_val,lb1_val,median_val,ub1_val,ub3_val]
    
        print(row_format.format(str(p)+':',
                                str('%1.2e' % mean_val),
                                str('%1.2e' % std_val),
                                str('%1.2e' % lb3_val),
                                str('%1.2e' % lb1_val),
                                str('%1.2e' % median_val),
                                str('%1.2e' % ub1_val),
                                str('%1.2e' % ub3_val)))

    return spec_params, samples



def chain_plot(spec_params,samples,truths=None,labels=None,xlog=True):

    ndim = spec_params.shape[0]
    
    fig, axes = plt.subplots(ndim, figsize=(10, ndim*2.5), sharex=True)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(np.arange(len(samples)),samples[:, :, i], alpha=0.1,marker='o')
        ax.set_xlim(1, len(samples))
        if truths != None:
            ax.plot([1,len(samples)],[truths[i],truths[i]],color='black')
        if labels != None:
            ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        if xlog == True:
            ax.set_xscale('log')
        
    axes[-1].set_xlabel("Iteration");


def corner_plot(spec_params,samples,truths=None,labels=None):

    nsamples, nwalkers, ndim = samples.shape
    
    samplesf = samples.reshape(nsamples*nwalkers,ndim)

    burnin = int(nsamples/2)
    
    sigma = 68.3
    fig = corner.corner(samplesf[burnin*nwalkers:,:],
                        labels=labels,
                        truths=truths,
                        quantiles=(50+sigma/np.array([-2,+2]))/100.,
                        show_titles=True,
                        bins=25,
                        fill_contours=True,
                        contourf_kwargs={"cmap": plt.cm.viridis, "colors":None},
                        levels=[1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)],
                        truth_color='orange')
    fig.set_size_inches(12,12)


def calc_posterior(ee,dee,matrix,e_lo,e_hi,e_model,spec_params,samples,fit_function):

    nsamples, nwalkers, ndim = samples.shape

    samplesf = samples.reshape(nsamples*nwalkers,ndim)
    
    n_e = len(ee)

    # how many samples to use for plotting and calculation of posterior model
    n_use = 250
    # data space
    n_plot_samples = nwalkers*n_use
    y_models = np.zeros((n_e,n_plot_samples))


    # where to evaluate model
    N_model = len(e_model)
    y_modelsm = np.zeros((N_model,n_plot_samples))

    last_x_samples = nsamples-n_use

    #print(n_walkers*last_x_samples,n_walkers*last_x_samples+n_plot_samples)

    for i in tqdm(range(nwalkers*last_x_samples,nwalkers*last_x_samples+n_plot_samples),'Loop over samples:'):
    #i = p - n_walkers*last_x_samples
        y_models[:,i-nwalkers*last_x_samples] = SPI_model_fit(samplesf[i,:],
                                                              ee,
                                                              None,
                                                              dee,
                                                              None,
                                                              matrix,
                                                              e_lo,
                                                              e_hi,
                                                              function=fit_function,
                                                              eval=True)


        y_modelsm[:,i-nwalkers*last_x_samples] = globals()[fit_function](e_model,samplesf[i,:])

    return y_models, y_modelsm


def plot_posterior(ee,y_models,edx=0,levels=[95.4,68.3],label=''):


    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(16,12),
                           gridspec_kw={'height_ratios':[4,1]})

    for level in levels:
        ax[0].fill_between(ee,
                           np.percentile(y_models, 50 - 0.5*level, axis=1 )*ee**edx,
                           np.percentile(y_models, 50 + 0.5*level, axis=1 )*ee**edx,
                           color='blue',alpha=0.3,step='mid',label=label)

    fit_model = np.median(y_models,axis=1)
    ax[0].step(ee,fit_model*ee**edx,linewidth=4,color='blue',where='mid',label=label+' (median model)')

    ax[0].legend()
    
    return fig, ax
    
