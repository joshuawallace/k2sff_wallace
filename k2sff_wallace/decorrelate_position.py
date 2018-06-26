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

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from joblib import Parallel, delayed
from sklearn.decomposition import PCA as PCA
from scipy.integrate import quad as quad
import bisect
from scipy.interpolate import splev
from scipy.interpolate import splrep
import copy

minimum_value = 1 # This minimum cadence value
maximum_value = 3856 # The maximum cadence value
n_jobs = 30 # The number of jobs to run in parallel
n_arclength_bins = 15 # Number of bins in arclength into which to bin the data

min_n_points_in_window_to_decorrelate = 10 # This is the minimum number of light curve points that need
# to be in a cadence window to run the decorrelation

#extra_to_skip = [192,193,194,195,196,197,198,199,200,201,204] # more cadences to skip b/c not conducive to Vanderburg analysis
#print("*********************************\n\nWe are still doing extra_to_skip\n\n********************************")

# Pickle file for positions over time
import calculate_posovertime_pickle
pos_over_time_pickle_filename = calculate_posovertime_pickle.pos_over_time_pickle_filename_touse

skip_first_cadence_division = True # Whether to skip the first cadence division, which contains a set of different pointings
start_cad = 50 # The starting cadence number, when skipping the first set of observations not pointed well

# Define where the cadences are to get split to run PCA and decorrelation on each chunk separately
cadence_number_divisions = [49+1, 570, 1084, 1600, 2154, 2570, 2995, 3419] # 8 divisions overall?
cadence_number_divisions.insert(0,minimum_value) # To get the first, original bounding value
cadence_number_divisions.append(maximum_value) # To get the last bounding value

# Frames to skip, and the cadences to use
to_skip = pickle.load( open( "../skip_cadences/to_skip.p", "rb" ) ) # normal to skip
#cadences = [val for val in range(minimum_value, maximum_value+1) if val not in to_skip] # Set of cadences without the extra_to_skip ones removed

# List of known RR Lyrae
known_rrlyrae = []
with open("../match_reference_to_gaia/list_of_rrlyrae.txt", "r") as f:
    known_rrlyrae = [val.strip("\n") for val in f.readlines()]

# List of stars significantly blended with the RR Lyrae, won't be able to do an accurate decorrelation on the light curves as they are now
probable_rrlyrae_blends = []
with open("../match_reference_to_gaia/list_of_rrlyrae_blends.txt", "r") as f:
    probable_rrlyrae_blends = [val.strip("\n") for val in f.readlines()]

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
            print "Number of apertures: " + str(len(aperture_line))
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


def calc_arclength(min_bound, max_bound, dy_dx):
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


def object_preparation(obj,cadence_number_divisions_):
    obj.calc_cadence_window_indices(cadence_number_divisions_)
    obj.calc_arclength()




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
        #self.cadence_window_id = None
        #self.bin_id = None
        self.cadence_window_indices = None
        self.arc_length = None

    """    
    def assign_cadence_window_id(self, cadence_window_edges):
        if min(self.cadence_nos) < min(cadence_window_edges):
            raise RuntimeError("The minimum cadence value is less than the minimum bin_edge value")
        if max(self.cadence_nos) > max(cadence_window_edges):
            raise RuntimeError("The maximum cadence value is greater than the maximum bin_edge value")
        temp = np.digitize(self.cadence_nos, cadence_window_edges, right=False)
        if min(temp) == 0:
            raise RuntimeError("np.digitize returned a value below the bottom of the bin edges")
        if max(temp) == len(cadence_window_edges):
            raise RuntimeError("np.digitize returned a value above the top of the bin edges")
        self.cadence_window_id = temp
    """

    def calc_cadence_window_indices(self, cadence_window_edges):
        if min(self.cadence_nos) < min(cadence_window_edges):
            raise RuntimeError("The minimum cadence value is less than the minimum bin_edge value")
        if max(self.cadence_nos) > max(cadence_window_edges):
            raise RuntimeError("The maximum cadence value is greater than the maximum bin_edge value")
        temp = []
        for val in cadence_window_edges:
            try:
                temp.append(self.cadence_nos.index(val))
            except ValueError:
                temp.append(bisect.bisect_right(self.cadence_nos, val))
        temp[-1] = temp[-1] + 1 # To get a number that will include the actual last position
        self.cadence_window_indices = temp

    """
    def assign_bins(self, bin_edges):
        if min(self.cadence_nos) < min(bin_edges):
            raise RuntimeError("The minimum cadence value is less than the minimum bin_edge value")
        if max(self.cadence_nos) > max(bin_edges):
            raise RuntimeError("The maximum cadence value is greater than the maximum bin_edge value")
        self.bin_id = np.digitize(self.cadence_nos, bin_edges, right=True)
    """

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
            #print pca_touse.fit_transform([[x,y] for x,y in zip(x_touse,y_touse)])
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
                arc_length_.append(calc_arclength(0., val, dy_dx))

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
        self.decorr_magnitudes = None

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

        # And redo the calculation of bin indices
        if len(filtered) == 0:
            self.object_info.cadence_window_indices = []
        else:
            self.object_info.calc_cadence_window_indices(cad_divisions)

    def pre_decorrelation_filtering():
        """
        This method filters low-frequency variations from the object's fluxes
        preparatory to the Vanderburg decorrelation.
        """
        breakpoint_time_length = 1.5

        """
        From Melinda Soares-Furtado
        ####Determine Bins for High-Pass Filter :1 Day Bins:#######################
        ####Cadence Intervals of 48 is about one day################################
        binsize=48
        beginningbin=Cadence[0]
        Reduced_Cadence=[]
        Reduced_Rel_Flux=[]
        for i in range(0,32):
            date=(Cadence[i]+binsize*(i+1)-beginningbin)/2+beginningbin
            Reduced_Cadence.append(date)
            tmp=[]
            weightstmp=[]
            for j in range(0,len(Cadence)):
                if (int(Cadence[j]) >= beginningbin) and  (int(Cadence[j]) < Cadence[i]+binsize*(i+1)):
                    tmp.append(flux[j])
                    weightstmp.append(Weights1[j])
            Reduced_Rel_Flux.append(np.average(tmp,weights=weightstmp))
            beginningbin=Cadence[i]+binsize*(i+1)
        SmoothedDates = np.linspace(Reduced_Cadence[0], max(Reduced_Cadence), 1000)
        tck4 = splrep(Reduced_Cadence,Reduced_Rel_Flux,k=1)
        Smoothed_Reduced_Rel_Flux = splev(SmoothedDates, tck4)   #breaks here
        Flux_highpass=[0]*len(flux)
        for t in range(0,len(flux)):
            indexval=find_nearest(SmoothedDates,Cadence[t])
            Flux_highpass[t]=flux[t]/Smoothed_Reduced_Rel_Flux[indexval]
        """
        t_min = self.object_info.BJD[0]
        t_max = self.object_info.BJD[-1]
        n_time_breakpoints = (t_max - t_min)/breakpoint_time_length
        n_time_breakpoints = int(round(n_time_breakpoints))
        breakpoints = np.linspace(t_min,t_max,num=n_time_breakpoints)
        breakpoints = breakpoints[1:-1]
        bspline_rep = splrep(x,y,t=breakpoints)
        splined_values = splev(x,bspline_rep)
        

        


def vanderburg_decorrelation(an_object_an_aperture):
    """
    Run the Vanderburg decorrelation for a one_object instance, an_object.
    Loops over all the apertures.
    """
    if not isinstance(an_object_an_aperture,one_object_one_aperture):
        raise RuntimeError("an_object is not a one_object instance")

    #print an_object_an_aperture.object_info.ID, " decorrelating"

    fluxes = np.power(10.,np.multiply(-0.4,an_object_an_aperture.magnitudes))
    #print "len fluxes: ",len(fluxes)
    fluxes_median = np.median(fluxes)
    fluxes_norm = [val/fluxes_median for val in fluxes]

    normalization_values = []
                    
    for i in range(len(an_object_an_aperture.object_info.cadence_window_indices)-1):
        if i == 0 and skip_first_cadence_division:
            # Skip any calculation for the cadences in the first cadence division (there shouldn't be
            # any observations left in this cadence division anyway)
            continue
            #normalization_values.extend( [1.] * an_object_an_aperture.object_info.cadence_window_indices[1])
        else:
            # Set the indices to be used
            lower_index = an_object_an_aperture.object_info.cadence_window_indices[i]
            upper_index = an_object_an_aperture.object_info.cadence_window_indices[i+1]

            # Extract which data to be used
            arc_length_to_use = an_object_an_aperture.object_info.arc_length[lower_index:upper_index]
            #if len(arc_length_to_use) < 10:
            #    print lower_index,upper_index, "   ", an_object_an_aperture.object_info.ID
            #    #print "here: ", arc_length_to_use
            # If there are no magnitudes in this cadence window to calculate on, skip any calculation
            if len(arc_length_to_use) == 0:
                continue
            if len(arc_length_to_use) < min_n_points_in_window_to_decorrelate:
                print an_object_an_aperture.object_info.ID, "   ", i, "  too few points to decorrelate against"
                normalization_values.extend([1.]*len(arc_length_to_use))
                continue
            fluxes_norm_to_use= fluxes_norm[lower_index:upper_index]

            # Now to bin up the data and then reject outliers,
            # then find mean of each bin.
            bins = np.linspace(min(arc_length_to_use), max(arc_length_to_use), n_arclength_bins+1)
            dbin = bins[1] - bins[0]
            bin_midpoints = [0.5*(bins[k+1] + bins[k]) for k in range(len(bins)-1)]
            which_bin = np.digitize(arc_length_to_use, bins, right=True)
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
                print "Zero first bin", len(binned_data[0]), an_object_an_aperture.object_info.ID, an_object_an_aperture.aperture_num

            if means_across_bins[-1] < 0.:
                print "Zero last bin", len(binned_data[-1]), an_object_an_aperture.object_info.ID, an_object_an_aperture.aperture_num

            # Now the values to decorrelate against
            decorrelation_values = np.interp(arc_length_to_use,bin_midpoints,means_across_bins,left=None,right=None)
            normalization_values.extend(decorrelation_values)


    if len(fluxes_norm) != len(normalization_values):
        print len(fluxes_norm)
        print len(normalization_values)
        raise RuntimeError("fluxes_norm and normalization_values do not have the same length")
    decorrelated_norm_lc = [fluxes_norm[k]/normalization_values[k] for k in range(len(fluxes_norm))]
    refluxed_decorrelated_lc = np.multiply(decorrelated_norm_lc, fluxes_median)
    remagnituded_decorrelated_lc = -2.5*np.log10(refluxed_decorrelated_lc)

    an_object_an_aperture.decorr_magnitudes = remagnituded_decorrelated_lc


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
                print "Comments did not have a '#' character prepended, adding now"
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

      



def main():
    
    # First try reading in the dicts that store the x and y positions
    try:
        with open(pos_over_time_pickle_filename, "rb") as f:
            x_pos_dict, y_pos_dict = pickle.load(f)
        print "Pickle file ", pos_over_time_pickle_filename ," found, and opened"

    except IOError:
        raise IOError("No Pickle file ",pos_over_time_pickle_filename," found.")


    # Now to store all the information for all the objects
    # Open a file to find some of the appropriate information
    
    gaiaID_list = np.genfromtxt("gaia_transformed_lists/gaia_transformed_1.txt",
                                                 dtype=str,usecols=0,unpack=True)
    #gaiaID_list = gaiaID_list[:1]#[:30]
    gaia_magnitudes, image_x, image_y = np.genfromtxt("gaia_transformed_lists/gaia_transformed_1.txt",
                                                 usecols=(3,6,7),unpack=True)
    all_objects = []
    #for ID in gaiaID_list[:3]:
    print "starting creating the objects"
    for i in range(len(gaiaID_list)):
        if i%200 == 0:
            print float(i)/float(len(gaiaID_list)) * 100.0, "% done making object instances"
        x_pos = x_pos_dict[gaiaID_list[i]]
        y_pos = y_pos_dict[gaiaID_list[i]]
        if len(x_pos) != len(y_pos):
            raise RuntimeError("The lengths of the x_pos and y_pos are different")

        BJD, cadence_nos = np.genfromtxt("../light_curves/grcollect_output/grcollect." + gaiaID_list[i] +\
                            ".grcollout", usecols=(1,2),dtype=(float,int),unpack=True)
        if len(x_pos) != len(cadence_nos) or len(BJD) != len(cadence_nos):
            print len(x_pos)
            print len(cadence_nos)
            print len(BJD)
            raise RuntimeError("The lengths of the cadence_nos or BJD are messed up")

        all_objects.append(one_object(gaiaID_list[i],cadence_nos,BJD,image_x[i],image_y[i],x_pos,y_pos))

    print "Now calling run_cadence_window_assignment_and_arclength_calc()"
    #arclength_output = Parallel(n_jobs=n_jobs)(delayed(run_cadence_window_assignment_and_arclength_calc)(obj) for obj in all_objects)
    print "------------"
    object_prep_output = Parallel(n_jobs=n_jobs, backend="threading")(delayed(object_preparation)(obj,cadence_number_divisions) for obj in all_objects)
    #for obj in all_objects:
    #    print obj.ID
    #    obj.calc_cadence_window_indices(cadence_number_divisions)
    #    obj.calc_arclength()


    #print all_objects[0].arc_length


    all_objects_with_lc = []
    print "\nNow starting to read in the light curves to the objects"
    for i in range(len(all_objects)):
        print "i: ",i, "     ", all_objects[i].ID
        if i%200 == 0:
            print float(i)/float(len(all_objects)) * 100.0, "% done making object instances"
        all_apertures_this_object = []
        for j in range(number_of_apertures):
            print "j: ",j
            mags_column_number = 13+5*j
            u,v,mags,errs = np.genfromtxt("../light_curves/grcollect_output/grcollect." +\
                              all_objects[i].ID + ".grcollout", usecols=(mags_column_number-2,
                                    mags_column_number-1,mags_column_number,mags_column_number+1),
                              missing_values='-',filling_values=float('nan'),unpack=True)
            
            this_object_this_aperture = one_object_one_aperture(all_objects[i],mags,errs,u,v,
                           cadence_number_divisions,j)

            all_apertures_this_object.append(this_object_this_aperture)

        all_objects_with_lc.append(all_apertures_this_object)


    #vanderburg_decorrelation(all_objects_with_lc[0][0])

    #vanderburg_decorrelation(all_objects_with_lc[0][0])
    #vanderburg_decorrelation(all_objects_with_lc[1][0])
    #vanderburg_decorrelation(all_objects_with_lc[2][0])
    print "starting the decorrelation calculation"
    decorrelation_output = Parallel(n_jobs=n_jobs, backend="threading")(delayed(vanderburg_decorrelation)(all_objects_with_lc[i][j]) for i in range(len(all_objects)) for j in range(number_of_apertures))
    #decorrelation_output = Parallel(n_jobs=n_jobs)(delayed(f__)(all_objects_with_lc[i][j]) for i in range(len(all_objects)) for j in range(number_of_apertures))
    #print all_objects_with_lc[0][0].magnitudes[900:915]
    #print all_objects_with_lc[0][0].decorr_magnitudes[900:915]


    #print all_objects[0].cadence_window_indices #[0, 49, 551, 1058, 1568, 2115, 2527, 2944, 3364, 3798]



    #print all_objects[0].arc_length

    objects_skipping = []

    print "\nStarting to save"
    for i in range(len(all_objects_with_lc)):
        if i%200 == 0:
            print float(i)/float(len(all_objects_with_lc)) * 100.0, "% done saving"
        for j in range(number_of_apertures):
            #print all_objects_with_lc[i][j].object_info.length
            if all_objects_with_lc[i][j].object_info.length == 0:
                print "skipping..."
                objects_skipping.append((all_objects_with_lc[i][j].object_info.ID, str(j)))
                continue
            save_lightcurve_to_file(all_objects_with_lc[i][j], 
                                    "test_decorr_output/" + all_objects_with_lc[i][j].object_info.ID +\
                                        "_" + str(j) + ".txt",
                                    "# " + all_objects_with_lc[i][j].object_info.ID + "    ap:" + str(j))

    with open("list_of_skipped_objects.txt","w") as f:
        f.write("# These are object/aperture combinations that had all nans for their magnitude values\n")
        f.write("# Gaia_ID  aperture_num\n")
        for obj in objects_skipping:
            f.write(obj[0] + "   " + obj[1] + "\n")

    
        
if __name__ == '__main__':
    main()
