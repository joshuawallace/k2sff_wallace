# Created Nov 15 2017 by JJW
#
# This code will take a given set light curves, decorrelate
# Vanderburg style, and then create the new light curves.
#
# This code is a redo of the previous decorrelate_position.py, which
# grew a bit bloated and disorganized by the end.
#
# Major changes in this code are:
#   - changed handling of frames to skip, as now the pipeline already skips
#        the extra frames that were necessary to skip for this code
#   - light curve information now stored as objects, for easier handling
#   - able to handle nan's for light curve values
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle as pickle
from joblib import Parallel, delayed
from sklearn.decomposition import PCA as PCA
from scipy.integrate import quad as quad
import bisect
from scipy.interpolate import splev
from scipy.interpolate import splrep
import copy

from signal import signal, SIGPIPE, SIG_DFL # As per https://stackoverflow.com/questions/14207708/ioerror-errno-32-broken-pipe-python
signal(SIGPIPE, SIG_DFL) 

# The minimum number of points needed in the time chunk to bother fitting a spline
num_point_to_fit_spline = 20


minimum_value = 1 # This minimum cadence value
maximum_value = 3856 # The maximum cadence value
n_jobs = 30 # The number of jobs to run in parallel
n_arclength_bins = 15 # Number of bins in arclength into which to bin the data

min_n_points_in_window_to_decorrelate = 10 # This is the minimum number of light curve points that need
# to be in a cadence window to run the decorrelation

# Pickle file for positions over time
import calculate_posovertime_pickle
pos_over_time_pickle_filename = calculate_posovertime_pickle.pos_over_time_pickle_filename_touse

skip_first_cadence_division = True # Whether to skip the first cadence division, which contains a set of different pointings
start_cad = 50 # The starting cadence number, when skipping the first set of observations not pointed well

# Define where the cadences are to get split to run PCA and decorrelation on each chunk separately
cadence_number_divisions = [49+1, 570, 1084, 1600, 2154, 2570, 2995, 3419] # 8 divisions overall?
cadence_number_divisions.insert(0,minimum_value) # To get the first, original bounding value
cadence_number_divisions.append(maximum_value) # To get the last bounding value

# Frames to skip
to_skip = pickle.load( open( "../skip_cadences/to_skip.p", "rb" ) ) # normal to skip


def get_number_of_apertures():
    """
    Read in the number of apertures from the photometry files
    """
    with open("../run_photometry/re_initial_photometry.sh","r") as f:
        lines = f.readlines()
    for line in lines:
        if "APERTURES=" in line and line[0] != '#':
            aperture_line = [item.strip("\n") for item in 
                             line.split("=")[1].split(",")]
            print ("Number of apertures: " + str(len(aperture_line)))
            break
    else:
        raise RuntimeError("Did not find the 'APERTURES' line in the file")
    return len(aperture_line)
number_of_apertures = get_number_of_apertures()


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Determine if two numbers are close in value (a kind of float equivalence test)
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def calc_arclength_integral(min_bound, max_bound, dy_dx):
    """
    This integrates the path length along y(x) given its  
    derivative, dy_dx, and the given min_bound and
    max_bound, then returns the path length.
    """
    integral = quad(lambda x: np.sqrt(1 + (dy_dx(x))**2), min_bound, max_bound)
    return integral[0]

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


def which_stddev_outliers_toexclude(y,splined_values,nsigma=3.):
    """
    This function determines which indices of y have
    |y - splined_values| < nsigma * sigma, after
    calculating sigma = stddev(y)
    """

    if len(y) != len(splined_values):
        print (len(y))
        print (len(splined_values))
        raise RuntimeError("y and splined_values are different lengths!")

    sigma = np.std(y)
    indices_of_outliers = [i for i in range(len(y)-1,-1,-1) if abs(y[i] - splined_values[i]) > nsigma*sigma]
    return indices_of_outliers


def check_breakpoints_have_data_between(times,breakpoints):
    ## Check that all the breakpoints have data in them, and fix if not
    if len(breakpoints) == 0:
        return []
    breakpoint_to_remove = []
    for i in range(len(breakpoints)):
        if i == 0:
            lower_val = times[0]
            upper_val = breakpoints[0]
        elif i == len(breakpoints)-1:
            lower_val = breakpoints[-1]
            upper_val = times[-1]
        else:
            lower_val = breakpoints[i-1]
            upper_val = breakpoints[i]
        for val in times: # If there's a value in between
            if lower_val < val < upper_val:
                break
        else: # If there's no value in between
            breakpoint_to_remove.append(i)
    return breakpoint_to_remove




def object_preparation(obj,cadence_number_divisions_):
    obj.calc_cadence_window_indices(cadence_number_divisions_)
    obj.calc_arclength()


num_steps = 8
def flatten_and_decorrelate(obj):
    i = 0
    previous_summeddiff = 0.
    while i < num_steps:
        i += 1
        if i > 1:
            obj.normalized_fluxes = list(obj.decorr_normalized_fluxes)
        obj.pre_decorrelation_filtering()
        vanderburg_decorrelation(obj,i)
        filtered = filter(lambda o: not np.isnan(o[0]) and not np.isnan(o[1]), zip(obj.previous_normalized_fluxes, obj.decorr_normalized_fluxes))
        if len(filtered) == 0:
            "zero-length filtered"
            break
        old, new = zip(*filtered)
        summed_diff = sum([abs(a-b) for a,b in zip(old,new)])

        summeddiff_reduced = summed_diff/float(len(old))
        if summeddiff_reduced <= 1e-5 or is_close(summeddiff_reduced,previous_summeddiff,rel_tol=2e-5, abs_tol=2e-5): #If converged
            #print (" num to converge: ", i)
            break
        else:
            previous_summeddiff = summeddiff_reduced
    #else:
    #    print (" **Wasn't able to converge in ", num_steps, " steps: ", obj.object_info.ID, " ", obj.aperture_num)
    return obj



class one_object:
    def __init__(self,ID_,cadence_nos_,BJD_,image_x_,image_y_,x_over_time_,y_over_time_):
        # Some checking
        if len(x_over_time_) != len(cadence_nos_) or len(y_over_time_) != len(cadence_nos_) or len(BJD_) != len(cadence_nos_):
            raise RuntimeError("The lengths of the x_pos or y_pos or BJD are different from the length of cadences in one_object")
        # Now creating the members of the class
        self.ID = ID_
        self.cadence_nos = list(cadence_nos_)
        self.x_over_time = list(x_over_time_)
        self.y_over_time = list(y_over_time_)
        self.BJD = list(BJD_)
        self.image_x = image_x_
        self.image_y = image_y_
        self.length = len(self.cadence_nos)
        self.cadence_window_indices = None
        self.arc_length = None


    def calc_cadence_window_indices(self, cadence_window_edges):
        if min(self.cadence_nos) < min(cadence_window_edges):
            raise RuntimeError("The minimum cadence value is less than the minimum bin_edge value")
        if max(self.cadence_nos) > max(cadence_window_edges):
            raise RuntimeError("The maximum cadence value is greater than the maximum bin_edge value")
        temp = []
        which_are_at_end = []
        for i in range(len(cadence_window_edges)):
            try:
                temp.append(self.cadence_nos.index(cadence_window_edges[i]))
            except ValueError:
                if cadence_window_edges[i] > self.cadence_nos[-1]: # If we are looking past the last available cadence number, make sure to not exceed the end of the array
                    temp.append(len(self.cadence_nos)-1)
                    which_are_at_end.append(i)
                else:
                    temp.append(bisect.bisect_right(self.cadence_nos, cadence_window_edges[i]))
        if len(which_are_at_end) > 0:
            for j in which_are_at_end:
                temp[j] = temp[j] + 1
        else:
            temp[-1] = temp[-1] + 1 # To get a number that will include the actual last position
        self.cadence_window_indices = temp




    def calc_arclength(self):
        if self.cadence_window_indices is None:
            raise RuntimeError("Need to set cadence window indices first")
        # Calculate the cadence *index* divisions
        arc_length_holder = []
        for i in range(len(self.cadence_window_indices)-1):
            x_touse = self.x_over_time[self.cadence_window_indices[i]:self.cadence_window_indices[i+1]]
            y_touse = self.y_over_time[self.cadence_window_indices[i]:self.cadence_window_indices[i+1]]

            # Use PCA to get the two main components of the data
            pca_touse = PCA(n_components=2, copy=True, svd_solver='auto', random_state=int(self.ID))
            transformed_positions = pca_touse.fit_transform([[x,y] for x,y in zip(x_touse,y_touse)])
            transformed_x, transformed_y = zip(*transformed_positions)

            # Now to fit a polynomial to the transformed positions, after re-centering
            minx = min(transformed_x)
            transformed_x = [val - minx for val in transformed_x]

            # Fit a fifth-degree polynomial as per Vanderburg (could possibly get away with lesser degree)
            pf = np.polyfit(transformed_x, transformed_y, 5)

            # Now define the polynomial's derivative for calculating the arclength.
            dy_dx = np.poly1d(np.polyder(pf))

            # Now calculate arc length
            arc_length_ = []
            for val in transformed_x:
                arc_length_.append(calc_arclength_integral(0., val, dy_dx))

            arc_length_holder.extend(arc_length_)

        if len(arc_length_holder) != len(self.cadence_nos):
            raise RuntimeError("The calculated arc lengths are not the same length (" +\
                                   str(len(arc_length_holder)) +\
                                   ") as the cadences (" + str(len(self.cadence_nos)) + ")")
        self.arc_length = arc_length_holder



class one_object_one_aperture:
    def __init__(self,one_object_instance,magnitudes,errors,centroid_x,centroid_y,
                 cad_divisions,aperture_number):
        if not isinstance(one_object_instance,one_object):
            raise RuntimeError("I was not given a one_object instance!")
        if len(magnitudes) != one_object_instance.length:
            raise RuntimeError("The magnitudes are not the same length as the one_object instance!")
        if len(magnitudes) != len(errors) or len(magnitudes) != len(centroid_x) or len(magnitudes) != len(centroid_y):
            raise RuntimeError("The magnitudes, errors, or centroids are not the same length!")
        self.object_info = copy.deepcopy(one_object_instance)
        self.aperture_num = aperture_number
        self.magnitudes = magnitudes
        self.errors = errors
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.decorr_flux = None
        self.decorr_normalized_fluxes = None
        self.decorr_magnitudes = None
        self.previous_normalized_fluxes = None # For storing the previous normalized fluxes to find when convergence occurs

        ##
        self.x_of_fit = None
        self.y_of_fit = None
        ##

        # Now, remove nan's from the light curve, and remove the corresponding values
        # (like cadence number) from the object info
        filtered = filter(lambda o: not np.isnan(o[0]), #and int(o[4]) not in extra_to_skip,
                                                        zip(self.magnitudes,self.errors,self.centroid_x,
                                                            self.centroid_y,self.object_info.cadence_nos,
                                                            self.object_info.BJD,
                                                            self.object_info.x_over_time,
                                                            self.object_info.y_over_time,
                                                            self.object_info.arc_length))
        if len(filtered) == 0:
            self.magnitudes = []
            self.errors = []
            self.centroid_x = []
            self.centroid_y = []
            self.object_info.cadence_nos = []
            self.object_info.BJD = []
            self.object_info.x_over_time = []
            self.object_info.y_over_time = []
            self.object_info.arc_length = []
        else:
            self.magnitudes,self.errors,self.centroid_x,self.centroid_y,self.object_info.cadence_nos,\
                self.object_info.BJD,self.object_info.x_over_time,\
                self.object_info.y_over_time,self.object_info.arc_length = zip(*filtered)

        # Now, if skip_first_cadence_division is True, remove all those points
        # from the object so we don't even consider them.
            if skip_first_cadence_division:
                filtered = filter(lambda o: o[4]>=start_cad, zip(self.magnitudes,self.errors,self.centroid_x,
                                                            self.centroid_y,self.object_info.cadence_nos,
                                                            self.object_info.BJD,
                                                            self.object_info.x_over_time,
                                                            self.object_info.y_over_time,
                                                            self.object_info.arc_length))
                self.magnitudes,self.errors,self.centroid_x,self.centroid_y,self.object_info.cadence_nos,\
                    self.object_info.BJD,self.object_info.x_over_time,\
                    self.object_info.y_over_time,self.object_info.arc_length = zip(*filtered)

        # Update length
        self.object_info.length = len(self.object_info.cadence_nos)

        # Now, convert magnitudes to flux and also calculate a median normalized flux
        self.fluxes = np.power(10.,np.multiply(-0.4,self.magnitudes))
        self.median_flux = np.median(self.fluxes)
        self.normalized_fluxes = self.fluxes/self.median_flux

        # And redo the calculation of bin indices
        if len(filtered) == 0:
            self.object_info.cadence_window_indices = []
        else:
            self.object_info.calc_cadence_window_indices(cad_divisions)

    def pre_decorrelation_filtering(self):
        """
        This method filters low-frequency variations from the object's fluxes
        preparatory to the Vanderburg decorrelation.
        """
        self.previous_normalized_fluxes = copy.deepcopy(self.normalized_fluxes) # Get the current normalized fluxes to save for later

        breakpoint_time_length = 1.5 # the length in days of the filter width
        
        x_of_fit_collect = []
        y_of_fit_collect = []
        values_to_normalize_by = []

        for ii in range(len(self.object_info.cadence_window_indices)-1):
            if ii == 0 and skip_first_cadence_division:
                # Skip any calculation for the cadences in the first cadence division (there shouldn't be
                # any observations left in this cadence division anyway)
                continue
            else:
                # Set the indices to be used
                lower_index = self.object_info.cadence_window_indices[ii]
                upper_index = self.object_info.cadence_window_indices[ii+1]
            if lower_index == upper_index: # If there is nothing to calculate in this time window
                continue
            
            # Figure out times to use for the spline fit (reaching beyond the cadence window)
            if not lower_index == 0: # If we're not on the smallest index
                lower_time = self.object_info.BJD[lower_index] - breakpoint_time_length/2.
                fitting_lower_index = bisect.bisect_left(self.object_info.BJD,lower_time)
                if fitting_lower_index > lower_index:
                    print (fitting_lower_index)
                    print (lower_index)
                    raise RuntimeError("The fitting lower index is higher than the nominal lower index")
            else:
                fitting_lower_index = lower_index
            if not upper_index == self.object_info.length: # If we're not on the largest index
                upper_time = self.object_info.BJD[upper_index - 1] + breakpoint_time_length/2.
                fitting_upper_index = bisect.bisect_left(self.object_info.BJD,upper_time)
                if fitting_upper_index < upper_index-1:
                    print (fitting_upper_index)
                    print (upper_index)
                    raise RuntimeError("The fitting upper index is higher than the nominal upper index")
            else:
                fitting_upper_index = upper_index
            # calculate breakpoint locations
            t_min = self.object_info.BJD[fitting_lower_index]
            t_max = self.object_info.BJD[fitting_upper_index-1]
            # If we don't have enough points to reliably fit a spline, skip this time chunk
            if (upper_index - lower_index) < num_point_to_fit_spline or (t_max - t_min) < breakpoint_time_length:
                values_to_normalize_by.extend([1.0]*(upper_index - lower_index))
                continue
            n_time_breakpoints = (t_max - t_min)/breakpoint_time_length
            n_time_breakpoints = int(round(n_time_breakpoints))
            breakpoints = np.linspace(t_min,t_max,num=n_time_breakpoints)
            breakpoints = breakpoints[1:-1]

            # put the fluxes and times in temporary arrays
            x = list(self.object_info.BJD[fitting_lower_index:fitting_upper_index+1])
            y = list(self.normalized_fluxes[fitting_lower_index:fitting_upper_index+1])

            ## Check that all the breakpoints have data in them, and fix if not
            breakpoint_to_remove = check_breakpoints_have_data_between(x,breakpoints)
            # Remove those values
            if breakpoint_to_remove:
                print ("&&&&&&&&&&&&&&&&")
                print ("We had some breakpoints to remove!!!")
                print ("")
                print (breakpoint_to_remove)
                print ("")
                print ("&&&&&&&&&&&&&&&&")
                breakpoint_to_remove.reverse()
                for val in breakpoint_to_remove:
                    breakpoints = np.delete(breakpoints,val)

            # iterate over spline fitting until converged (i.e., number of new points to exclude is zero in a particular iteration)
            num_points_excluded = 1
            num_loops = 0
            while num_points_excluded != 0:
                # create and fit the spline
                bspline_rep = splrep(x,y,t=breakpoints)
                splined_values = splev(x,bspline_rep)

                # figure out which ones to sigma-exclude, and then remove them
                which_ones_to_remove = which_stddev_outliers_toexclude(y,splined_values,nsigma=3.)
                for i in which_ones_to_remove:
                    _ = x.pop(i)
                    _ = y.pop(i)
                num_points_excluded = len(which_ones_to_remove)
                if len(breakpoints) > 1:
                    if num_points_excluded >= 1 and (breakpoints[0] < x[0] or breakpoints[-1] > x[-1]): #If some points are excluded and the breakpoints are no longer okay, recalculate breakpoints
                        if breakpoints[0] < x[0]:
                            t_min = x[0] - breakpoint_time_length/2.
                        if breakpoints[-1] > x[-1]:
                            t_max = x[-1] + breakpoint_time_length/2.
                        n_time_breakpoints = (t_max - t_min)/breakpoint_time_length
                        n_time_breakpoints = int(round(n_time_breakpoints))
                        breakpoints = np.linspace(t_min,t_max,num=n_time_breakpoints)
                        breakpoints = breakpoints[1:-1]
                        if breakpoints[0] < x[0] or breakpoints[-1] > x[-1]: # Double check to make sure we didn't screw things up
                            print (breakpoints)
                            print (x[0])
                            print (x[-1])
                            raise RuntimeError("Messed up the breakpoint re-calculation")
                else:
                    print ("length of breakpoints was zero, " + self.object_info.ID + " " +  str(self.aperture_num) + " " + str(num_loops))
                if num_points_excluded >= 1:
                    ## Check that all the breakpoints have data in them, and fix if not
                    breakpoint_to_remove = check_breakpoints_have_data_between(x,breakpoints)
                    # Remove those values
                    if breakpoint_to_remove:
                        print ("&&&&&&&&&&&&&&&&")
                        print ("We had some breakpoints to remove!!!")
                        print ("")
                        print (breakpoint_to_remove)
                        print ("")
                        print ("&&&&&&&&&&&&&&&&")
                        breakpoint_to_remove.reverse()
                        for val in breakpoint_to_remove:
                            breakpoints = np.delete(breakpoints,val)
                ############
                num_loops += 1
            if num_loops > 20:
                print ("Object " + self.object_info.ID + " had a very much larger than normal number of loops in the spline fitting: " + str(num_loops))

            ##
            xp = np.linspace(self.object_info.BJD[lower_index],self.object_info.BJD[upper_index-1],500)
            yp = splev(xp,bspline_rep)
            x_of_fit_collect.append(xp)
            y_of_fit_collect.append(yp)
            ##    

            # Save the values to normalize by
            splev(self.object_info.BJD,bspline_rep)
            values_to_normalize_by.extend(splev(self.object_info.BJD[lower_index:upper_index],bspline_rep))
        # Now, normalize by the calculated spline
        if len(self.normalized_fluxes) != len(values_to_normalize_by):
            print (len(self.normalized_fluxes))
            print (len(values_to_normalize_by))
            raise RuntimeError("The splined values to normalize by were not the same length as the values to normalize")
        temp = [val[0]/val[1] for val in zip(self.normalized_fluxes,values_to_normalize_by)]
        self.normalized_fluxes = temp

        # Save the values for the spline fit to use to plot later
        self.x_of_fit = x_of_fit_collect
        self.y_of_fit = y_of_fit_collect

        # Double check that it's still the correct length
        if len(self.normalized_fluxes) != self.object_info.length:
            print (len(self.normalized_fluxes))
            print (self.object_info.length)
            raise RuntimeError("After normalizing fluxes by spline fit, the lengths changed.")
        


def vanderburg_decorrelation(an_object_an_aperture,iternum):
    """
    Run the Vanderburg decorrelation for a one_object instance, an_object.
    Loops over all the apertures.
    """
    if not isinstance(an_object_an_aperture,one_object_one_aperture):
        raise RuntimeError("an_object is not a one_object instance")

    normalization_values = []
                    
    for i in range(len(an_object_an_aperture.object_info.cadence_window_indices)-1):
        if i == 0 and skip_first_cadence_division:
            # Skip any calculation for the cadences in the first cadence division (there shouldn't be
            # any observations left in this cadence division anyway)
            continue
        else:
            # Set the indices to be used
            lower_index = an_object_an_aperture.object_info.cadence_window_indices[i]
            upper_index = an_object_an_aperture.object_info.cadence_window_indices[i+1]

            # Extract which data to be used
            arc_length_to_use = an_object_an_aperture.object_info.arc_length[lower_index:upper_index]
            # If there are no magnitudes in this cadence window to calculate on, skip any calculation
            if len(arc_length_to_use) == 0:
                continue
            if len(arc_length_to_use) < min_n_points_in_window_to_decorrelate:
                print (an_object_an_aperture.object_info.ID + "   " + str(i) + "  too few points to decorrelate against")
                normalization_values.extend([1.]*len(arc_length_to_use))
                continue
            fluxes_norm_to_use= an_object_an_aperture.normalized_fluxes[lower_index:upper_index]

            # Now to bin up the data and then reject outliers,
            # then find mean of each bin.
            bins = np.linspace(min(arc_length_to_use), max(arc_length_to_use), n_arclength_bins+1)
            dbin = bins[1] - bins[0]
            bin_midpoints = [0.5*(bins[k+1] + bins[k]) for k in range(len(bins)-1)]
            which_bin = np.digitize(arc_length_to_use, bins, right=True)
            
            pos_of_zero_values = np.where(which_bin == 0)[0]
            if len(pos_of_zero_values) == 0:
                raise RuntimeError("The minimum value didn't fall outside the bound?")
            if len(pos_of_zero_values) > 1:
                raise RuntimeError("There is more than 1 point that fell below the bounds.")
            which_bin[pos_of_zero_values[0]] = 1
            
            binned_data = [np.array(fluxes_norm_to_use)[which_bin == k] for k in range(1, len(bins))]
            binned_arclength = [np.array(arc_length_to_use)[which_bin == k] for k in range(1, len(bins))]

            # Now, for those bins with no values in them, let's skip them and their corresponding midpoints
            bins_with_no_values = np.where(np.array([len(l) for l in binned_data]) == 0)[0].tolist()

            for index in sorted(bins_with_no_values, reverse=True):
                #_ = binned_data_removed_outliers.pop(index)
                _ = binned_data.pop(index)
                _ = bin_midpoints.pop(index)

            binned_data_removed_outliers = [reject_outliers_stddev(l) for l in binned_data]

            # Now, take the means of the non-empty bins
            means_across_bins = [np.mean(l) for l in binned_data_removed_outliers]


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

            if means_across_bins[0] < 0.:
                print ("Zero first bin       " + str(len(binned_data[0])) + "  " + an_object_an_aperture.object_info.ID + "  " + stran_object_an_aperture.aperture_num))
                print ("iter num: " + str(iternum))
                print (binned_data[0])

            if means_across_bins[-1] < 0.:
                print ("Zero last bin    " + str(len(binned_data[-1])) + "  " + an_object_an_aperture.object_info.ID + "  " + str(an_object_an_aperture.aperture_num))
                print ("iter num: " +  str(iternum))
                print (binned_data[-1])

            # Now the values to decorrelate against
            decorrelation_values = np.interp(arc_length_to_use,bin_midpoints,means_across_bins,left=None,right=None)
            normalization_values.extend(decorrelation_values)


    if len(an_object_an_aperture.normalized_fluxes) != len(normalization_values):
        print (len(an_object_an_aperture.normalized_fluxes))
        print (len(normalization_values))
        raise RuntimeError("the object's normalized fluxes and normalization_values do not have the same length")
    decorrelated_norm_lc = [an_object_an_aperture.normalized_fluxes[k]/normalization_values[k] for k in range(len(an_object_an_aperture.normalized_fluxes))]
    refluxed_decorrelated_lc = np.multiply(decorrelated_norm_lc, an_object_an_aperture.median_flux)
    remagnituded_decorrelated_lc = -2.5*np.log10(refluxed_decorrelated_lc)

    an_object_an_aperture.decorr_flux = refluxed_decorrelated_lc
    an_object_an_aperture.decorr_normalized_fluxes = decorrelated_norm_lc
    an_object_an_aperture.decorr_magnitudes = remagnituded_decorrelated_lc

    return copy.deepcopy(an_object_an_aperture)


def save_lightcurve_to_file(an_object_ap_instance, output_filename,comments=None):
    """
    This function will take the light curve file found at input_lightcurve_file
    (which is a path to the the light curve file), which is in the 
    fiphot lightcurve format, and to output_filename (a path for the output 
    file) writes a modified version of the input_lightcurve_file that
    has the decorrelated lightcurve values in an_object_ap_instance.

    It also skips the line numbers corresponding to lines_to_skip.
    """

    with open(output_filename,"w") as f:
        if comments:
            comment_touse = comments
            if comment_touse[0] != '#':
                print ("Comments did not have a '#' character prepended, adding now")
                comment_touse = "#" + comment_touse

            f.write(comment_touse + "\n")
        
        for i in range(an_object_ap_instance.object_info.length):
            str_to_write = "%-15.8f %4d %11.7f %11.7f %9.5f %9.5f %10.5f %10.5f\n" %\
                (an_object_ap_instance.object_info.BJD[i],
                 an_object_ap_instance.object_info.cadence_nos[i],
                 an_object_ap_instance.object_info.image_x,
                 an_object_ap_instance.object_info.image_y,
                 an_object_ap_instance.centroid_x[i],
                 an_object_ap_instance.centroid_y[i],
                 an_object_ap_instance.decorr_magnitudes[i],
                 an_object_ap_instance.errors[i])
            f.write(str_to_write)

      



def one_chunk(chunk_num,num_chunks,gaiaID_list,image_x,image_y,x_pos_dict,y_pos_dict):
    print ("  Running chunk " + str(chunk_num) + " out of " + str(num_chunks))

    #index_min = chunk_num*chunksize
    #if chunk_num == num_chunks:
    #    index_max = len(gaiaID_list_full)
    #else:
    #    index_max = (chunk_num+1)*chunksize

    all_objects = []
    #gaiaID_list = gaiaID_list_full[index_min:index_max]
    #image_x = image_x_full[index_min:index_max]
    #image_y = image_y_full[index_min:index_max]
    print ("starting creating the objects for chunk " + str(chunk_num) + " out of " + str(num_chunks))
    for i in range(len(gaiaID_list)):
        #if gaiaID_list[i] not in objects_wanted:
        #    continue
        x_pos = x_pos_dict[gaiaID_list[i]]
        y_pos = y_pos_dict[gaiaID_list[i]]
        if len(x_pos) != len(y_pos):
            raise RuntimeError("The lengths of the x_pos and y_pos are different")

        BJD, cadence_nos = np.genfromtxt("../light_curves/grcollect_output/grcollect." + gaiaID_list[i] +\
                            ".grcollout", usecols=(1,2),dtype=(float,int),unpack=True)
        if len(x_pos) != len(cadence_nos) or len(BJD) != len(cadence_nos):
            print (len(x_pos))
            print (len(cadence_nos))
            print (len(BJD))
            raise RuntimeError("The lengths of the cadence_nos or BJD are messed up")

        all_objects.append(one_object(gaiaID_list[i],cadence_nos,BJD,image_x[i],image_y[i],x_pos,y_pos))

    #print ("Now calling run_cadence_window_assignment_and_arclength_calc(), chunk ", chunk_num)
    #print ("------------")
    for obj in all_objects:
        object_preparation(obj,cadence_number_divisions)

    # Now to loop over all the apertures for each object and get all the light curve information
    all_objects_with_lc = []
    print ("\nNow starting to read in the light curves to the objects, chunk " + str(chunk_num))
    for i in range(len(all_objects)):
        #print ("i: ",i, "     ", object_prep_output[i].ID)
        #if i%200 == 0:
        #    print (float(i)/float(len(object_prep_output)) * 100.0, "% done making object instances")
        all_apertures_this_object = []
        for j in range(number_of_apertures):
            #print ("j: ",j)
            mags_column_number = 13+5*j
            u,v,mags,errs = np.genfromtxt("../light_curves/grcollect_output/grcollect." +\
                              all_objects[i].ID + ".grcollout", usecols=(mags_column_number-2,
                                    mags_column_number-1,mags_column_number,mags_column_number+1),
                              missing_values='-',filling_values=float('nan'),unpack=True)

            this_object_this_aperture = one_object_one_aperture(all_objects[i],mags,errs,u,v,
                           cadence_number_divisions,j)

            all_objects_with_lc.append(this_object_this_aperture)


    print ("\nNow running the flattening and decorrelation, chunk " + str(chunk_num) + " of " + str(num_chunks))
    #decorrelation_output = Parallel(n_jobs=n_jobs)(delayed(flatten_and_decorrelate)(obj) for obj in all_objects_with_lc)
    decorrelation_output = []
    for obj in all_objects_with_lc:
        flatten_and_decorrelate(obj)




    print ("\nStarting to save, chunk " + str(chunk_num) + " of " + str(num_chunks))
    objects_skipping = []
    for i in range(len(all_objects_with_lc)):
        #if i%400 == 0:
        #    print (float(i)/float(len(all_objects_with_lc)) * 100.0, "% done saving")
        if all_objects_with_lc[i].object_info.length == 0:
            #print ("skipping...")
            objects_skipping.append((all_objects_with_lc[i].object_info.ID, str(all_objects_with_lc[i].aperture_num)))
            continue
        save_lightcurve_to_file(all_objects_with_lc[i], 
                                "decorr_output_iter/" + all_objects_with_lc[i].object_info.ID +\
                                    "_" + str(all_objects_with_lc[i].aperture_num) + ".txt",
                                "# " + all_objects_with_lc[i].object_info.ID + "    ap:" + str(all_objects_with_lc[i].aperture_num))

    return objects_skipping

def main():

    chunksize = 40 # How many objects to do at a time

    #objects_wanted = []
    #with open("../light_curves/full_lightcurves.txt","r") as f: # get objects with full light curves
    #    text_file = f.readlines()
    #    for i in range(1,len(text_file)): #1 to start is to skip comment line
    #        objects_wanted.append(text_file[i].split()[0])
    #objects_wanted = ['6045478043228461440']


    
    # First try reading in the dicts that store the x and y positions
    try:
        with open(pos_over_time_pickle_filename, "rb") as f:
            x_pos_dict, y_pos_dict = pickle.load(f)
        print ("Pickle file " + str(pos_over_time_pickle_filename) + " found, and opened")

    except IOError:
        raise IOError("No Pickle file ",pos_over_time_pickle_filename," found.")


    # Now to store all the information for all the objects
    # Open a file to find some of the appropriate information
    
    gaiaID_list_full = np.genfromtxt("gaia_transformed_lists/gaia_transformed_1.txt",
                                                 dtype=str,usecols=0,unpack=True)
    image_x_full, image_y_full = np.genfromtxt("gaia_transformed_lists/gaia_transformed_1.txt",
                                                 usecols=(6,7),unpack=True)
    
    ## Now here, we will loop over a number of different chunks, to make memory manageable
    num_chunks = len(gaiaID_list_full)//chunksize

    #gaiaID_list = gaiaID_list_full[index_min:index_max]
    #image_x = image_x_full[index_min:index_max]
    #image_y = image_y_full[index_min:index_max]

    list_of_iiis = range(num_chunks + 1)
    all_objects_skipping = Parallel(n_jobs=n_jobs)(delayed(one_chunk)(iii,num_chunks,
                                                                      gaiaID_list_full[iii*chunksize:(iii+1)*chunksize],
                                                                      image_x_full[iii*chunksize:(iii+1)*chunksize],
                                                                      image_y_full[iii*chunksize:(iii+1)*chunksize],
                                                                      x_pos_dict,y_pos_dict) for iii in list_of_iiis)
    print (all_objects_skipping)
    quit()

    with open("list_of_skipped_objects_iter.txt","w") as f:
        f.write("# These are object/aperture combinations that had all nans for their magnitude values\n")
        f.write("# Gaia_ID  aperture_num\n")
        for obj in objects_skipping:
            f.write(obj[0] + "   " + obj[1] + "\n")

    
        
if __name__ == '__main__':
    main()
