#!/usr/bin/env python 

#
# Added valid "FRB" beam-masks for the Parkes Multibeam receiver as the new co-incident RFI test
#

import sys, math
import numpy as np
from math import sin, pi

class Classifier(object):
    def __init__(self):
        self.nbeams      = 13
        self.snr_cut     = 10.0
        self.members_cut = 200 
        self.nbeams_cut  = 13
        self.dm_min      = 200
        self.dm_max      = 500
        self.filter_min  = 6
        self.filter_max  = 11

    def is_hidden(self, cand):
        return ( (cand['snr'] < self.snr_cut) |
                 (cand['filter'] < self.filter_min) | (cand['filter'] > self.filter_max) |
                 (cand['nbeams'] < self.nbeams_cut) ) 

    def is_noise(self, cand):
        return cand['members'] < self.members_cut

    # test if candidate is galactic
    def is_out_dmrange(self, cand):
      return ((cand['dm'] < self.dm_min) | (cand['dm'] > self.dm_max))

    def is_not_adjacent(self, cand, epoch, half_time):
      return (cand['time'] > epoch + half_time) | (cand['time'] < epoch - half_time)

    # count the maximum time
    def min_time(self, cand):
        return np.amin(cand['time'])

    # count the maximum time
    def max_time(self, cand):
        return np.amax(cand['time'])

class TextOutput(object):
    def __init__(self):
        self.dm_base = 1.0
        self.snr_min = 6.0

    def print_html(self, data):
        if len(data['valid']) > 0:
            sys.stdout.write("<table width='100%' border=1 cellpadding=4px cellspacing=4px>\n")
            sys.stdout.write("<tr><th align=left>SNR</th><th align=left>Time</th><th align=left>DM</th><th align=left>Filter [ms]</th><th align=left>Beam</th></tr>\n")
            for (i, item) in enumerate(data['valid']['snr']):
                sys.stdout.write ("<tr>" + \
                                  "<td>" + str(data['valid']['max_snr'][i]) + "</td>" + \
                                  "<td>" + str(data['valid']['time'][i]) + "</td>" + \
                                  "<td>" + str(data['valid']['dm'][i]) + "</td>" + \
                                  "<td>" + str(0.064 * (2 **data['valid']['filter'][i])) + "</td>" + \
                                  "<td>" + str(data['valid']['prim_beam'][i]+1) + "</td>" + \
                                  "</tr>\n")
            sys.stdout.write("</table>\n")

    def print_text(self, data):
        
        cand_type = 'valid'
    
        if len(data[cand_type]) > 0:

          # get indicies list for sorting via time
          sorted_indices = [i[0] for i in sorted(enumerate(data[cand_type]['time']), key=lambda x:x[1])]

          sys.stdout.write ( "SNR\tTIME\tSAMP\tDM\tFILTER\tPRI_BEAM\n")
          for i in sorted_indices:
                sys.stdout.write (str(data[cand_type]['max_snr'][i]) + "\t" + \
                                  str(data[cand_type]['time'][i]) + "\t" + \
                                  str(data[cand_type]['samp_idx'][i]) + "\t" + \
                                  str(data[cand_type]['dm'][i]) + "\t" + \
                                  str(data[cand_type]['filter'][i]) + "\t" + \
                                  str(data[cand_type]['prim_beam'][i]+1) + \
                                  "\n")

    def print_xml(self, data):
        # get indicie list for sorting via snr
        snr_sorted_indices = [i[0] for i in sorted(enumerate(data['valid']['max_snr']), key=lambda x:x[1],reverse=True)]

        cand_i = 0
        for i in snr_sorted_indices:
            cand_i += 1
            sys.stdout.write ("<candidate snr='" + str(data['valid']['max_snr'][i]) + \
                                       "' time='" + str(data['valid']['time'][i]) + \
                                       "' dm='" + str(data['valid']['dm'][i]) + \
                                       "' samp_idx='" + str(data['valid']['samp_idx'][i]) + \
                                       "' filter='" + str(data['valid']['filter'][i]) + \
                                       "' prim_beam='" + str(data['valid']['prim_beam'][i] + 1) + "'/>\n")




if __name__ == "__main__":
    import argparse
    import Gnuplot
    
    parser = argparse.ArgumentParser(description="Detects Perytons in candidates file")
    parser.add_argument('-cands_file', default="all_candidates.dat")

    parser.add_argument('-snr_cut', type=float, default=10)
    parser.add_argument('-filter_min', type=int, default=7)
    parser.add_argument('-filter_max', type=int, default=10)

    parser.add_argument('-max_cands_per_sec', type=float, default=2)
    parser.add_argument('-cand_list_xml', action="store_true")
    parser.add_argument('-cand_list_html', action="store_true")
    parser.add_argument('-verbose', action="store_true")
    args = parser.parse_args()
    
    max_cands_per_second = args.max_cands_per_sec
    filename = args.cands_file
    verbose = args.verbose
    cand_list_xml = args.cand_list_xml
    cand_list_html = args.cand_list_html
    
    # Load candidates from all_candidates file
    all_cands = \
        np.loadtxt(filename,
                   dtype={'names': ('snr','samp_idx','time','filter',
                                    'dm_trial','dm','members','begin','end',
                                    'nbeams','beam_mask','prim_beam',
                                    'max_snr','beam'),
                          'formats': ('f4', 'i4', 'f4', 'i4',
                                      'i4', 'f4', 'i4', 'i4', 'i4',
                                      'i4', 'i4', 'i4',
                                      'f4', 'i4')})

    # Adjust for 0-based indexing
    all_cands['prim_beam'] -= 1
    all_cands['beam'] -= 1

    if verbose:
      sys.stderr.write ("Loaded %i candidates\n" % len(all_cands))
    
    classifier = Classifier()
    classifier.snr_cut = args.snr_cut
    classifier.filter_min = args.filter_min
    classifier.filter_max = args.filter_max
    
    # Filter candidates based on classifications
    if verbose:
      sys.stderr.write ("Classifying candidates...\n")

    categories = {}

    is_hidden      = classifier.is_hidden(all_cands)
    is_noise       = classifier.is_noise(all_cands)
    is_out_dmrange = classifier.is_out_dmrange(all_cands)
    is_valid       = (is_hidden == False) & (is_noise == False) & (is_out_dmrange == False)

    categories["hidden"]      = all_cands[is_hidden]
    categories["noise"]       = all_cands[(is_hidden == False) & is_noise]
    categories["out_dmrange"] = all_cands[(is_hidden == False) & (is_noise == False) & is_out_dmrange]
    categories["valid"]       = all_cands[is_valid]

    pre_valid = len(categories["valid"])

    # for valid events, check the event rate around the time of the event
    if len(categories['valid']) > 0 & False:
      min_time = classifier.min_time(all_cands)
      max_time = classifier.max_time(all_cands)

      # look in a 8 second window around the event for excessive RFI
      event_time = 6

      for (i, item) in reversed(list(enumerate(categories['valid']))):

        epoch = item['time']
        half_time = event_time / 2.0
        new_time = 0

        if (epoch - half_time) < min_time:
          new_time += (epoch - min_time)
        else:
          new_time += half_time

        if (epoch + half_time) > max_time:
          new_time += (max_time - epoch)
        else:
          new_time += half_time

        half_time = new_time / 2.0

        is_not_adjacent = (is_noise == False) & (is_out_dmrange == False) & classifier.is_not_adjacent(all_cands, epoch, half_time)
        is_valid = (is_noise == False) & (is_out_dmrange == False) & (is_not_adjacent == False)

        event_sum = float(np.count_nonzero(is_valid)) - 12
        cands_per_second = event_sum / new_time

        if cands_per_second < 0:
          cands_per_second = 0
          if verbose:
            sys.stderr.write ( "cands_per_second = %f \n" % ( cands_per_second )) 
        if verbose:
          sys.stderr.write ( "cands_per_second around %f was %f [max = %f]\n" % ( item['time'], cands_per_second, max_cands_per_second )) 
        if cands_per_second >= max_cands_per_second:
          categories['valid'] = np.delete(categories['valid'], i, axis=0)

    rfi_storm = pre_valid - len(categories["valid"])

    if verbose:
      sys.stderr.write ( "Classified %i as hidden \n" % len(categories["hidden"]))
      sys.stderr.write ( "           %i as noise spikes\n" % len(categories["noise"]))
      sys.stderr.write ( "           %i as out of DM range \n" % len(categories["out_dmrange"]))
      sys.stderr.write ( "           %i as RFI storm\n" % rfi_storm)
      sys.stderr.write ( "           %i as valid Peryton candidates\n" % len(categories["valid"]))

    text_output = TextOutput()

    if cand_list_xml:
      text_output.print_xml(categories)
    elif cand_list_html:
      text_output.print_html(categories)
    else:
      text_output.print_text(categories)

    if verbose:
      sys.stderr.write ( "Done\n")

