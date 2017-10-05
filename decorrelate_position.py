# Created Sept 8 2017 by JJW
#
# This code will take a given set light curves, decorrelate
# Vanderburg style, and then create the new light curves.
#
# It does so over all the apertures, not just specific
# ones determined by the magnitude of the object.
#
# It also runs parallel

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from joblib import Parallel, delayed
from sklearn.decomposition import PCA as PCA
from scipy.integrate import quad as quad
import warnings


# So that the warnings about negative values in the log trigger an exception
#warnings.filterwarnings("error", "invalid value encountered in log10", RuntimeWarning)

def aperture_num_to_use(mag):
    """
    Returns an aperture number to use based on the input magnitude.
    The aperture numbers to use as a function of magnitude were determined
    empirically based on the rms vs. magnitude plots for each aperture
    """
    if mag < 13.5:
        return 6
    elif mag < 14.5:
        return 5
    elif mag < 15.5:
        return 2
    else:
        return 1


minimum_value = 1
maximum_value = 3856

n_jobs = 30

n_arclength_bins = 15 # Number of bins in arclength into which to bin the data
undefined = -1000. # a value to use if a point is outside the interpolation range

# Pickle file for positions over time
pos_over_time_pickle_filename = "pos_over_time_all.p"

skip_first_cadence_division = True # Whether to skip the first cadence division, which contains a set of different pointings

# Define where the cadences are to get split to run PCA and decorrelation on each chunk separately
cadence_number_divisions = [49+1, 570, 1084, 1600, 2154, 2570, 2995, 3419] # 8 divisions overall?
cadence_number_divisions.insert(0,minimum_value) # To get the first, original bounding value
cadence_number_divisions.append(maximum_value) # To get the last bounding value


# Frames to skip
to_skip = pickle.load( open( "../skip_cadences/to_skip.p", "rb" ) ) # normal to skip
extra_to_skip = [192,193,194,195,196,197,198,199,200,201,204] # more cadences to skip b/c not conducive to Vanderburg analysis
extra_to_skip_reverse_sorted = sorted(extra_to_skip, reverse=True) # reverse sort for later .pop() use
cadences_orig = [val for val in range(minimum_value, maximum_value+1) if val not in to_skip] # Set of cadences without the extra_to_skip ones removed
cadences = [val for val in cadences_orig if val not in extra_to_skip] # So the cadences are reflective of the extra removed things


# List of known RR Lyrae
known_rrlyrae = []
with open("../match_reference_to_gaia/list_of_rrlyrae.txt", "r") as f:
    known_rrlyrae = [val.strip("\n") for val in f.readlines()]

probable_rrlyrae_blends = []
with open("../match_reference_to_gaia/list_of_rrlyrae_blends.txt", "r") as f:
    probable_rrlyrae_blends = [val.strip("\n") for val in f.readlines()]


def get_number_of_apertures():
    with open("../run_photometry/re_initial_photometry.sh","r") as f:
        lines = f.readlines()
    for line in lines:
        if "APERTURES=" in line and line[0] != '#':
            aperture_line = [item.strip("\n") for item in 
                             line.split("=")[1].split(",")]
            print "Number of apertures: " + str(len(aperture_line))
            break
    else:
        raise RuntimeError("Did not find the 'APERTURES' line in the file")
    return len(aperture_line)

number_of_apertures = get_number_of_apertures()

def get_positions_over_time(indices_wanted):
    """
    Extracts the positions as a function of time for the objects corresponding to the
    input indices_wanted (appropriate indices must be determined ahead of time
    """
    x_positions_over_time = [list([]) for _ in range(len(indices_wanted))]
    y_positions_over_time = [list([]) for _ in range(len(indices_wanted))]


    for i in cadences:
        if i % 200 == 0:
            print round(float(i)/float(maximum_value)*100.,2),"% of the way done getting positions"
        data = np.genfromtxt("gaia_transformed_lists/gaia_transformed_" + str(i) + ".txt",
                         usecols=(-4,-3,-2,-1))
        for j in range(len(indices_wanted)):
            x_positions_over_time[j].append(data[j][2] - data[j][0])
            y_positions_over_time[j].append(data[j][3] - data[j][1])

    return (x_positions_over_time, y_positions_over_time)


arcsec_per_pixel = 3.98
def plot_relativemagnitude_vs_arclength(relfluxes, fitted_value, arclengths, output_path, single_bin=None, mag=None):
    """
    This plots the relative magnitude vs. arclength plots, so we can see how those things go.
    """
    #print ":::::::::::::"
    #print len(relfluxes)
    #print len(fitted_value)
    #print len(arclengths)
    #print ":::::::::::::"
    for i in range(len(relfluxes)):
        plt.scatter(np.multiply(arclengths[i],arcsec_per_pixel), relfluxes[i], color='black', s=7.5)
    # Plot first few values to see ifff.....
    plt.scatter(np.multiply(arclengths[0][:7],arcsec_per_pixel), relfluxes[0][:7], color='lime', s=9.5)
    for i in range(len(fitted_value)):
        plt.scatter(np.multiply(arclengths[i],arcsec_per_pixel), fitted_value[i], color='orange', s=2.5)
    
    plt.xlim(left=0)
    plt.xlabel("Arclength (arcseconds)")
    plt.ylabel("Relative Flux")

    if single_bin != None:
        plt.figtext(0.8, 0.15, "s.b.: " + str(single_bin),size=16)
    if mag != None:
        plt.title("Magnitude: " + str(mag))

    if "png" in output_path:
        plt.savefig(output_path,dpi=400)
    else:
        plt.savefig(output_path)
    plt.close()



def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def calc_arclength(min_bound, max_bound, dy_dx):
    """
    This integrates the path length along y(x) given its  
    derivative, dy_dx, and the given min_bound and
    max_bound, then returns the path length.
    """
    integral = quad(lambda x: np.sqrt(1 + (dy_dx(x))**2), min_bound, max_bound)
    return integral[0]


def reject_outliers_MAD(data, m = 3.): # Thanks to https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    This performs a median absolute deviation removal of outliers
    in the set of numbers in data, removing things that are
    more than m times beyond the median absolute deviation
    """
    if len(data) in [0,1]: # If an empty list, nothing to remove, or a 1-length list, no MAD to calculate
        return data
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    #print len(data)
    return data[s<m]

def reject_outliers_stddev(data, n_sigma = 3.): # Thanks to https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    This performs a standard deviation removal of outliers
    in the set of numbers in data, removing things that are
    more than m times beyond the standard deviation
    """
    if len(data) in [0,1]: # If an empty list, nothing to remove, or a 1-length list, no stddev to calculate
        return data
    stddev = np.std(data)
    if is_close(stddev, 0., abs_tol=1.e-9):
        # Check to make sure all the values are close to the same; if so, return all of them again
        any_different = False
        for i in range(len(data)-1):
            for j in range(i+1, len(data)):
                if not is_close(data[i],data[j]):
                    any_different = True
                    break
            if any_different:
                break
        if not any_different:
            return data
        else:
            raise RuntimeError("Standard deviation is close to zero, but still the data values are different from each other")

    return data[abs(data - np.mean(data)) < n_sigma * stddev]
    

def get_cadence_index_divisions():
    # Figure out the indices in the data corresponding to the cadence numbers used in the divisions 
    cadence_index_divisions = []
    for num in cadence_number_divisions:
        try:
            cadence_index_divisions.append(cadences.index(num)) # find the index for the cadence number
        except ValueError: # didn't find the cadence number
            if num in to_skip: # if not found because this cadence number was skipped
                cadence_index_divisions.append(cadences.index(num+1)) # try the next cadence number
            else:
                raise RuntimeError("Missing index not in to_skip")
    if len(cadence_index_divisions) != len(cadence_number_divisions): # if we didn't get the right number of cadence_index_divisions
        raise RuntimeError("Did not find all indices for cadence divisions")
    cadence_index_divisions[-1] = cadence_index_divisions[-1] + 1 # To get a number that will include the actual last position
    return cadence_index_divisions



def vanderburg_decorrelation(i, objects_wanted, cadence_index_divisions, x_positions_this_i, y_positions_this_i):
    #warnings.filterwarnings("once", "Negative flux associated with a probable RR Lyrae blend", RuntimeWarning)

    print i
    if i % 20 == 0:
        print float(i)/float(len(objects_wanted))*100., "% done"

    if i % 1 == 0:
        plot_relmag_v_arclength = True # Whether to generate a relative mag vs. arclength plot for this
        single_bin = 0
        relflux_for_plotting = []
        flux_corrected_for_plotting = []
        mag_median = 0.
    else:
        plot_relmag_v_arclength = False
        return [[0,0],[0,0]]


    all_raw_lightcurves_this_i = []
    all_decorrelated_lightcurves_this_i = []


    # Do some preliminary arclength work, that doesn't need to be repeated over the apertures
    x_positions_to_use = list(x_positions_this_i) #copy them all over
    y_positions_to_use = list(y_positions_this_i)
    # for each value in the extra_to_skip list, pop out the corresponding value in these lists
    for val in extra_to_skip_reverse_sorted:
        index_to_pop = cadences_orig.index(val) # the index corresponding to this cadence
        _ = x_positions_to_use.pop(index_to_pop)
        _ = y_positions_to_use.pop(index_to_pop)

    arc_length_over_cadence_groups = []

    beginning_group = 0
    if skip_first_cadence_division: beginning_group = 1 # If we are to skip the first group of cadences
    for j in range(beginning_group+1,len(cadence_index_divisions)):
        # use PCA to get the two main components of the data
        pca_touse = PCA(n_components=2, copy=True, svd_solver='auto', random_state=1*i)
        transformed_positions = pca_touse.fit_transform([[x,y] for x,y in zip(x_positions_to_use[cadence_index_divisions[j-1]:cadence_index_divisions[j]],y_positions_to_use[cadence_index_divisions[j-1]:cadence_index_divisions[j]])])
        transformed_x, transformed_y = zip(*transformed_positions)

        # Now to fit a polynomial to the transformed positions, after re-centering
        minx = min(transformed_x)
        transformed_x = [val - minx for val in transformed_x]

        # Fit a fifth-degree polynomial as per Vanderburg (could possibly get away with lesser degree)
        pf = np.polyfit(transformed_x, transformed_y, 5)

        # Now define the polynomial's derivative for calculating the arclength.
        dy_dx = np.poly1d(np.polyder(pf))

        # Now calculate arc length
        arc_length = []
        for val in transformed_x:
            arc_length.append(calc_arclength(0., val, dy_dx))

        #print "len transformed positions"
        #print len(transformed_positions)
        #print max(arc_length)
        #for kk in range(len(arc_length)):
        #        if arc_length[kk] > 1.256:
        #            print "it did it!"
        #            print i, j, kk, arc_length[kk] 

        arc_length_over_cadence_groups.append(arc_length)


    #not_notified_yet = True

    for a in range(number_of_apertures):
    # get the magnitudes as a function of time
        lc_mags = np.genfromtxt("../light_curves/grcollect_output/grcollect." +\
                         str(objects_wanted[i]) + ".grcollout", 
                         usecols=11+3*(a) )


        # Now get the positions and magnitudes to use, skipping over the cadences around cadence 200
        # that have positions way off from the others
        lc_mags_to_use     = list(lc_mags)
        lc_fluxes_to_use   = list(np.power(10.,np.multiply(-0.4,lc_mags_to_use)))
        fluxes_median = np.nanmedian(lc_fluxes_to_use)
        lc_fluxes_norm_to_use = [val/fluxes_median for val in lc_fluxes_to_use]
        # for each value in the extra_to_skip list, pop out the corresponding value in these lists
        for val in extra_to_skip_reverse_sorted:
            index_to_pop = cadences_orig.index(val) # the index corresponding to this cadence
            _ = lc_mags_to_use.pop(index_to_pop)
            _ = lc_fluxes_to_use.pop(index_to_pop)
            _ = lc_fluxes_norm_to_use.pop(index_to_pop)
        all_raw_lightcurves_this_i.append(lc_mags_to_use)

        normalization_values = [] # For collecting the normalization values over the different cadence divisions

        beginning_group = 0
        if skip_first_cadence_division: beginning_group = 1 # If we are to skip the first group of cadences
        for j in range(beginning_group+1,len(cadence_index_divisions)):
            arc_length_to_use = arc_length_over_cadence_groups[j - (beginning_group + 1)]

            # get correct lc_fluxes_norm_to_use
            lc_fluxes_norm_this_caddiv = lc_fluxes_norm_to_use[cadence_index_divisions[j-1]:cadence_index_divisions[j]]


            if plot_relmag_v_arclength and a==3:
                relflux_for_plotting.append(lc_fluxes_norm_this_caddiv)
                median_mag = np.nanmedian(lc_mags_to_use)

            """
            # Reject clear outliers that could mess up things if left in, if they're all in the same bin for example
            mean_val = np.mean(lc_fluxes_norm_this_caddiv)
            stddev_val = np.std(lc_fluxes_norm_this_caddiv)
            indices_to_reject = [ ll for ll in range(len(lc_fluxes_norm_this_caddiv)) if abs(lc_fluxes_norm_this_caddiv[ll] - np.mean(lc_fluxes_norm_this_caddiv)) > 5. * stddev_val]
            
            if len(indices_to_reject) > 0:
                "Number ID of object that removed some flux values: ", i, " and had ", len(indices_to_reject), " removed."
                print i,a,j
                print indices_to_reject
                print len(arc_length_to_use)
                warnings.warn("Object " + str(i) + " had " + str(len(indices_to_reject)) + " flux values removed from the whole cadence set", RuntimeWarning)
                for index in sorted(indices_to_reject, reverse=True):
                    _ = lc_fluxes_norm_this_caddiv.pop(index)
                    _ = arc_length_to_use.pop(index)
            """


            # Now to bin up the data and then reject outliers,
            # then find mean of each bin.
            bins = np.linspace(min(arc_length_to_use), max(arc_length_to_use), n_arclength_bins+1)
            dbin = bins[1] - bins[0]
            bin_midpoints = [0.5*(bins[k+1] + bins[k]) for k in range(len(bins)-1)]
            which_bin = np.digitize(arc_length_to_use, bins, right=True)
            binned_data = [np.array(lc_fluxes_norm_this_caddiv)[which_bin == k] for k in range(1, len(bins))]
            binned_arclength = [np.array(arc_length_to_use)[which_bin == k] for k in range(1, len(bins))]


            # Now, for those bins with no values in them, let's skip them and their corresponding midpoints
            bins_with_no_values = np.where(np.array([len(l) for l in binned_data]) == 0)[0].tolist()

            for index in sorted(bins_with_no_values, reverse=True):
                #_ = binned_data_removed_outliers.pop(index)
                _ = binned_data.pop(index)
                _ = bin_midpoints.pop(index)

            binned_data_removed_outliers = [reject_outliers_stddev(l) for l in binned_data]

            
            
            
            """
            for k in range(len(binned_data)):
                a = len(binned_data[k]) - len(binned_data_removed_outliers[k])
                if float(a)/float(len(binned_data[k])) >= .2:
                    print "For", i, " ", a, " ", j, " ", k, " there are ", float(a)/float(len(binned_data[k]))*100., "% removed, total number ", a," out of", len(binned_data[k])
                    print binned_data[k]
            """


            # Now, take the means of the non-empty bins
            means_across_bins = [np.mean(l) for l in binned_data_removed_outliers]


            """
            # Now, we'll want to find bins which are themselves outliers and remove those from consideration to avoid messing up the interpolation
            bins_to_ignore = [ll for ll in range(len(means_across_bins)) if abs(means_across_bins[ll] - np.mean(means_across_bins)) > 5. * np.std(means_across_bins)]

            if j == 5 and a == 15:
                print means_across_bins
                print np.std(means_across_bins)
                print abs(means_across_bins[-2] - np.mean(means_across_bins))

            if len(bins_to_ignore) > 0:
                # We'll want to ignore this bin for the interpolation, and then also make sure to include the proper normalization value of 1 for the bin contents
                print "wowee!"
            """


            """if len(binned_data[0]) == 1:
                bin_midpoints.insert(0,bins[0])
            bin_midpoints.append(bins[-1])
            try:
                means_across_bins.insert(0, np.poly1d(np.polyfit(binned_arclength[0],binned_data[0],1))(bins[0]))
            except Warning:
                print "insert 0", len(binned_arclength[0]), len(binned_data[0])
            try:
                means_across_bins.append(np.poly1d(np.polyfit(binned_arclength[-1],binned_data[-1],1))(bins[-1]))
            except Warning:
                print "append", len(binned_arclength[-1]), len(binned_data[-1])"""

            # Now make points for the first and last bin edges to cover all the data
            # First, if the first and last bins have more than one point in them, do the following
            if len(binned_data_removed_outliers[0]) > 1:
                bin_midpoints.insert(0,bins[0])
                means_across_bins.insert(0,means_across_bins[0] - .5*(means_across_bins[1] - means_across_bins[0]))
            else:
                bin_midpoints.insert(0,bins[0])
                means_across_bins.insert(0,binned_data_removed_outliers[0][0])
                bin_midpoints[1] = bin_midpoints[2] - .5*dbin # Use the .5*dbin distance away from other midpoints instead of calculated bin value, in case the bin value reflects a bin that was empty and thus already removed from consideration
                means_across_bins[1] = means_across_bins[2] - .5*(means_across_bins[3] - means_across_bins[2])

            if len(binned_data_removed_outliers[-1]) > 1:
                bin_midpoints.append(bins[-1])
                means_across_bins.append(means_across_bins[-1] + .5*(means_across_bins[-1] - means_across_bins[-2]))
            else:
                bin_midpoints.append(bins[-1])
                means_across_bins.append(binned_data_removed_outliers[-1][0])
                bin_midpoints[-2] = bin_midpoints[-3] + .5*dbin # Use the .5*dbin distance away from other midpoints instead of calculated bin value, in case the bin value reflects a bin that was empty and thus already removed from consideration
                means_across_bins[-2] = means_across_bins[-3] + .5*(means_across_bins[-3] - means_across_bins[-4])

            """
            if  len(binned_data[0]) == 1 and a == 3:
                #print "len 1 first bin", i
                pass
            if means_across_bins[0] < 0.:
                print "Zero first bin", len(binned_data[0]), i, a, j

            if  len(binned_data[-1]) == 1 and a == 3:
                #print "len 1 last bin", i
                pass
            """
                
            if means_across_bins[-1] < 0.:
                print "Zero last bin", len(binned_data[-1]), i, a, j
                print [len(l) for l in binned_data]
                print [len(l) for l in binned_data_removed_outliers]

                    
            #if single_bin == 0 and a==3:
            #    print "--------------"
            #    print "--------------"
            #    print " It's zero!!!"
            #    print i
                
                
            """
            if  (len(binned_data[0]) == 1 or len(binned_data[-1]) == 1) and a == 3:
                single_bin += 1
                print "now plotting"
                plt.scatter(arc_length_to_use, [1]*len(arc_length_to_use), s=3)
                for val in bins:
                    plt.plot([val,val],[.9,1.1],color='black')
                plt.savefig("pdf/arclength_" + str(i) + "_" + str(j) + ".pdf")
                plt.close()
            """


            decorrelation_values = np.interp(arc_length_to_use,bin_midpoints,means_across_bins,left=undefined,right=undefined)
            """
            if len(indices_to_reject) > 0:
                decorrelation_values = list(decorrelation_values)
                for index in sorted(indices_to_reject):
                    decorrelation_values.insert(index, 1.0)

                print [x for _,x in sorted(zip(arc_length_to_use,decorrelation_values))][-10:]"""
                    
            if plot_relmag_v_arclength and a==3:
                flux_corrected_for_plotting.append(decorrelation_values)

            #nan_in_decorr_value = np.argwhere(np.isnan(decorrelation_values)).tolist()
            # Will want to divide the light curve by this value
            normalization_values.extend(decorrelation_values)

            #if means_across_bins[-1] < 0. and a == 10 and j==2:
            plot_relativemagnitude_vs_arclength([lc_fluxes_norm_this_caddiv], [decorrelation_values], [arc_length_to_use], "pdf/zerobin_check_onthings_18.pdf")


        if skip_first_cadence_division:
            decorrelated_norm_lc = lc_fluxes_norm_to_use[:cadence_index_divisions[1]] + [lc_fluxes_norm_to_use[k+cadence_index_divisions[1]]/normalization_values[k] for k in range(len(lc_fluxes_norm_to_use[cadence_index_divisions[1]:]))]
        else:
            decorrelated_norm_lc = [lc_fluxes_norm_to_use[k]/normalization_values[k] for k in range(len(lc_fluxes_norm_to_use))]

        if plot_relmag_v_arclength and a==3:
            plot_relativemagnitude_vs_arclength(relflux_for_plotting, flux_corrected_for_plotting, arc_length_over_cadence_groups, "pdf/relflux_v_arclength_" + str(i)+ ".png", single_bin=single_bin, mag=median_mag)


        # Convert back to pseudo-flux from the normalized value
        refluxed_decorrelated_lc = np.multiply(decorrelated_norm_lc,fluxes_median)

        """if np.any(refluxed_decorrelated_lc < 0.):
            print "----"
            print i
            print a
            temp = np.argwhere(refluxed_decorrelated_lc < 0.)[0]
            print temp
            #print refluxed_decorrelated_lc(np.argwhere(refluxed_decorrelated_lc < 0.))
            for val in temp:
                print refluxed_decorrelated_lc[val]
                print decorrelated_norm_lc[val]
                print lc_fluxes_norm_to_use[val]
                print normalization_values[val-cadence_index_divisions[1]]
            print "----" """


        """if np.any(refluxed_decorrelated_lc < 0.):
            print "--------------"
            print "Negative value here!!!"
            print i
            fff = len([val for val in refluxed_decorrelated_lc if val < 0.])
            print fff
            try:
                print len(fff)
            except TypeError:
                print "1"
            for val in fff:
                print refluxed_decorrelated_lc.tolist().index(val)
            print "-------------"
        """

        # Now convert back to magnitudes
        remagnituded_decorrelated_lc = -2.5*np.log10(refluxed_decorrelated_lc)
        """try:
            remagnituded_decorrelated_lc = -2.5*np.log10(refluxed_decorrelated_lc)
        except RuntimeWarning:
            print objects_wanted[i]
            if not str(objects_wanted[i]) in known_rrlyrae:
                if str(objects_wanted[i]) in probable_rrlyrae_blends:
                    if  not_notified_yet:
                        print "Number ID of object with negative value and probable RR Lyrae blend: ", i
                        not_notified_yet = False
                    warnings.warn("Negative flux associated with a probable RR Lyrae blend", RuntimeWarning)
                else:
                    print "Number ID of object with warning: ", i
                    raise
            else:
                warnings.warn("RR Lyrae with some negative flux values after correction", RuntimeWarning)
                
        """

        #And append to a list
        all_decorrelated_lightcurves_this_i.append(remagnituded_decorrelated_lc.tolist())

        #"../light_curves/grcollect_output/grcollect." + str(objects_wanted[i]) + ".grcollout"

    # Save light curves
    save_lightcurve_to_file(all_decorrelated_lightcurves_this_i, 
                            "../light_curves/grcollect_output/grcollect." +\
                             str(objects_wanted[i]) + ".grcollout",
                             "decorr_output/decorr_lc_" + str(objects_wanted[i]) + ".dat",
                             [cadences_orig.index(val) for val in extra_to_skip])

    #print "end ", i
    return [all_raw_lightcurves_this_i, all_decorrelated_lightcurves_this_i]



num_initial_characters = 91 # How many characters it takes to get to the 
            # first magnitude value
num_characters_per_aperture = 26 # How many characters to get to the 
            # next magnitude value
num_characters_flux_value = 8 # The number of characters for the value itself
w=8 # The maximum width of a number to print out


def save_lightcurve_to_file(decorrelated_lightcurves, input_lightcurve_file, output_filename, lines_to_skip):
    """
    This function will take the light curve file found at input_lightcurve_file
    (which is a path to the the light curve file), which is in the 
    fiphot lightcurve format, and to output_filename (a path for the output 
    file) writes a modified version of the input_lightcurve_file that
    has the values in decorrelated_lightcurves, which itself is in the format
    of lightcurves generated by vanderburg_decorrelation().

    It also skips the line numbers corresponding to lines_to_skip.

    This function is probably easiest to run from inside
    vanderburg_decorrelation(), and in the initial incarnation
    of this code that is where it is called.
    """
    #print "saving light curve yay!"


    with open(input_lightcurve_file,"r") as f:
        lines = f.readlines()
    
    if len([l for l in lines if l[0] != '#']) != len(decorrelated_lightcurves[0]) + len(lines_to_skip):
        print len([l for l in lines if l[0] != '#'])
        print len(decorrelated_lightcurves[0])
        raise RuntimeError("Number of lines in light curve file and this light curve calculation don't match up.")

    header_lines = [line for line in lines if line[0] == '#']

    lines_to_print = []

    i_to_use = [val for val in range(len(lines) - len(header_lines)) if val not in lines_to_skip]

    for i in range(len(i_to_use)):
        temp_line = lines[i_to_use[i] + len(header_lines)]
        for j in range(number_of_apertures):
            num_to_print = decorrelated_lightcurves[j][i]
            if not np.isnan(num_to_print):
                p=w-len(str(int(num_to_print)))-1 # Number of digits left for decimals
                str_num_to_print = "{:{w}.{p}f}".format(num_to_print, w=w, p=p)

                position = num_initial_characters + j*num_characters_per_aperture


                """if temp_line[position + num_characters_flux_value] != ' ':
                    print j
                    print temp_line[position + num_characters_flux_value-40:position + num_characters_flux_value+40]
                    print temp_line[position + num_characters_flux_value-4:position + num_characters_flux_value+4]
                    raise RuntimeError("Character following flux value not a space, instead was a " + str(temp_line[position + num_characters_flux_value]))"""
                temp_line = temp_line[:position] + str_num_to_print + \
                    temp_line[position + num_characters_flux_value:]
            else:
                str_num_to_print = '-'.rjust(num_characters_flux_value)

        lines_to_print.append(temp_line)
            
    if len(lines_to_print) != len(lines) - len(header_lines) - len(lines_to_skip):
        print len(lines_to_print)
        print len(lines) - len(header_lines)
        raise RuntimeError("Did not end up with enough lines after processing " +
                       "all\n the lines to be same as original number")

    #print "saving process started"
    with open(output_filename, "w") as f:
        for line in header_lines:
            f.write(line)
        for line in lines_to_print:
            f.write(line)
    #print "saving process ended"

    pass # Just a visual marker to close the loop and end the function



def main():
    # Figure out the indices in the data corresponding to the cadence numbers used in the divisions 
    cadence_index_divisions = get_cadence_index_divisions()
    print "Cadence index divisions: ", cadence_index_divisions


    # Figure out which the indexes of the objects we want to plot, with object
    # names initially in ../light_curves/full_lightcurves.txt
    # but planned to be extended to all

    indices_wanted = []
    magnitudes_of_wanted = []
    objects_wanted = []
    with open("../light_curves/full_lightcurves.txt","r") as f: # get objects with full light curves
        text_file = f.readlines()
        for i in range(1,len(text_file)): #1 to start is to skip comment line
            objects_wanted.append(int(text_file[i].split()[0]))

    sample_list = np.genfromtxt("gaia_transformed_lists/gaia_transformed_1.txt",
                                dtype=np.int64,usecols=0,unpack=True)

    their_magnitudes = np.genfromtxt("gaia_transformed_lists/gaia_transformed_1.txt",
                            usecols=3,unpack=True)


    for i in range(len(objects_wanted)):
        for j in range(len(sample_list)):
            if sample_list[j] == objects_wanted[i]:
                indices_wanted.append(j)
                magnitudes_of_wanted.append(their_magnitudes[j])
                break

        else:
            raise RuntimeError("Didn't find a match for this object, ", objects_wanted[i])


    ### Try to read in the pickle file; generate it if doesn't exist
    try:
        with open(pos_over_time_pickle_filename, "rb") as f:
            x_positions_over_time, y_positions_over_time = pickle.load(f)
        print "Pickle file ", pos_over_time_pickle_filename ," found, now opening"


    except IOError:
        print "No Pickle file ",pos_over_time_pickle_filename," found, now generating."
        x_positions_over_time, y_positions_over_time = get_positions_over_time(indices_wanted)
        with open(pos_over_time_pickle_filename,"wb") as f:
            pickle.dump((x_positions_over_time, y_positions_over_time),f)



    ### Now decorrelate each light curve and plot up
    print "Starting parallel portion"
    decorrelation_output = Parallel(n_jobs=n_jobs)(delayed(vanderburg_decorrelation)(i, objects_wanted, cadence_index_divisions, x_positions_over_time[i], y_positions_over_time[i]) for i in range(len(objects_wanted)))
    #decorrelation_output = Parallel(n_jobs=n_jobs)(delayed(vanderburg_decorrelation)(i, objects_wanted, cadence_index_divisions, x_positions_over_time[i], y_positions_over_time[i]) for i in [18])

    #i = 18
    #output = vanderburg_decorrelation(i, objects_wanted, cadence_index_divisions, x_positions_over_time[i], y_positions_over_time[i]) 


    print "Now pulling out the raw light curves"
    all_raw_lightcurves = [val[0] for val in decorrelation_output]
    print "Now pulling out the decorrelated light curves"
    all_decorrelated_lightcurves = [val[1] for val in decorrelation_output]



    # determine which aperture to use for plotting
    print "now plotting"
    magnitudes_of_wanted = magnitudes_of_wanted
    which_one = [aperture_num_to_use(val) for val in magnitudes_of_wanted]


    raw_to_use = [all_raw_lightcurves[i][which_one[i]-1] for i in range(len(all_raw_lightcurves))]
    decorrelated_to_use = [all_decorrelated_lightcurves[i][which_one[i]-1] for i in range(len(all_decorrelated_lightcurves))]

    rms_raw = [np.std(l[cadence_index_divisions[1]:]) for l in raw_to_use]
    rms_decorrelated = [np.std(l[cadence_index_divisions[1]:]) for l in decorrelated_to_use]
    MAD_raw = [np.median(np.absolute(np.subtract(l[cadence_index_divisions[1]:], np.median(l[cadence_index_divisions[1]:])))) for l in raw_to_use]
    MAD_decorrelated = [np.median(np.absolute(np.subtract(l[cadence_index_divisions[1]:], np.median(l[cadence_index_divisions[1]:])))) for l in decorrelated_to_use]


    print "median RMS:"
    for a in range(number_of_apertures):
        print a,": ", np.nanmedian( [np.std(all_decorrelated_lightcurves[i][a][cadence_index_divisions[1]:]) for i in range(len(all_decorrelated_lightcurves)) ])

    print "median MAD:"
    for a in range(number_of_apertures):
        print a,": ", np.nanmedian( [np.median(np.absolute(np.subtract(all_decorrelated_lightcurves[i][a][cadence_index_divisions[1]:], np.median(all_decorrelated_lightcurves[i][a][cadence_index_divisions[1]:])) ))for i in range(len(all_decorrelated_lightcurves)) ])
    

    
    plt.scatter(magnitudes_of_wanted, rms_decorrelated, color='red',s=4,label="decorr")
    plt.scatter(magnitudes_of_wanted, rms_raw, color='blue',s=4,label="raw")
    plt.xlabel("Gaia magnitude")
    plt.ylabel("RMS")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig("pdf/vanderburg_decorr_rms_4.pdf")
    plt.close()

    plt.scatter(magnitudes_of_wanted, MAD_decorrelated, color='red',s=4,label="decorr")
    plt.scatter(magnitudes_of_wanted, MAD_raw, color='blue',s=4,label="raw")
    plt.xlabel("Gaia magnitude")
    plt.ylabel("MAD")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig("pdf/vanderburg_decorr_mad_4.pdf")
    plt.close()
    



if __name__ == '__main__':
    main()
