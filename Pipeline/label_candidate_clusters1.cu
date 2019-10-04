/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/label_candidate_clusters.h"
#include "hd/are_coincident.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>

#define SNR_BASED_CLUSTERING

__device__ unsigned int d_counter;

// Finds the root of a chain of equivalent labels
//   E.g., 3->1, 4->3, 8->4, 5->8 => [1,3,4,5,8]->1
// TODO: It would be quite interesting to study the behaviour of this
//         algorithm/implementation in more detail.
template<typename T>
struct trace_equivalency_chain {
	T* new_labels;
	trace_equivalency_chain(T* new_labels_) : new_labels(new_labels_) {}
	inline /*__host__*/ __device__
	void operator()(unsigned int old_label) const {
		T cur_label = old_label;
		while( new_labels[cur_label] != cur_label ) {
			cur_label = new_labels[cur_label];
			//new_labels[old_label] = cur_label;
			// TESTING TODO: See if/how this varies if we write
			//                 new_labels[old_label] each iteration vs.
			//                 only at the end (see commented line below).
			//               It appears to make only 10-20% difference
			atomicAdd(&d_counter, 1);
		}
		new_labels[old_label] = cur_label;
		
	}
};

struct cluster_functor {
	hd_size  count;
	const hd_size* d_samp_inds;
	const hd_size* d_begins;
	const hd_size* d_ends;
	const hd_size* d_filters;
	const hd_size* d_dms;
#ifdef SNR_BASED_CLUSTERING
  const hd_float* d_peaks;
#endif

	hd_size* d_labels;
	hd_size  time_tol;
	hd_size  filter_tol;
	hd_size  dm_tol;
	
	cluster_functor(hd_size count_,
	                const hd_size* d_samp_inds_,
	                const hd_size* d_begins_, const hd_size* d_ends_,
	                const hd_size* d_filters_, const hd_size* d_dms_,
#ifdef SNR_BASED_CLUSTERING
                  const hd_float* d_peaks_,
#endif
	                hd_size* d_labels_,
	                hd_size time_tol_, hd_size filter_tol_, hd_size dm_tol_)
		: count(count_),
		  d_samp_inds(d_samp_inds_),
		  d_begins(d_begins_), d_ends(d_ends_),
		  d_filters(d_filters_), d_dms(d_dms_),
#ifdef SNR_BASED_CLUSTERING
      d_peaks(d_peaks_),
#endif
		  d_labels(d_labels_),
		  time_tol(time_tol_), filter_tol(filter_tol_), dm_tol(dm_tol_) {}
	
	inline __host__ __device__
	void operator()(unsigned int i) {
		hd_size samp_i   = d_samp_inds[i];
		hd_size begin_i  = d_begins[i];
		hd_size end_i    = d_ends[i];
		hd_size filter_i = d_filters[i];
		hd_size dm_i     = d_dms[i];
		// TODO: This would be much faster using shared mem like in nbody
		for( unsigned int j=0; j<count; ++j ) {
			if( j == i ) {
				continue;
			}
			hd_size samp_j   = d_samp_inds[j];
			hd_size begin_j  = d_begins[j];
			hd_size end_j    = d_ends[j];
			hd_size filter_j = d_filters[j];
			hd_size dm_j     = d_dms[j];
			if( are_coincident(samp_i, samp_j,
			                   begin_i, begin_j,
			                   end_i, end_j,
			                   filter_i, filter_j,
			                   dm_i, dm_j,
			                   time_tol, filter_tol, dm_tol) ) {
#ifdef SNR_BASED_CLUSTERING
        if (d_peaks[i] < d_peaks[j])
          d_labels[i] = d_labels[j];
#else
				// Re-label as the minimum of the two
				d_labels[i] = min((int)d_labels[i], (int)d_labels[j]);
#endif
			}
		}
	}
};

// Finds components of the given list that are connected in time, filter and DM
// Note: merge_dist is the distance in time up to which components are connected
// Note: Merge distances in filter and DM space are currently fixed at 1
// TODO: Consider re-naming the *_count args to *_max
hd_error label_candidate_clusters(hd_size            count,
                                  ConstRawCandidates d_cands,
                                  hd_size            time_tol,
                                  hd_size            filter_tol,
                                  hd_size            dm_tol,
                                  hd_size*           d_labels,
                                  hd_size*           label_count)
{
	using thrust::make_counting_iterator;
	
	thrust::device_ptr<hd_size> d_labels_begin(d_labels);
	thrust::sequence(d_labels_begin, d_labels_begin+count);
	
	// This just does a brute-force O(N^2) search for neighbours and
	//   re-labels as the minimum label over neighbours.
	thrust::for_each(make_counting_iterator<unsigned int>(0),
	                 make_counting_iterator<unsigned int>(count),
	                 cluster_functor(count,
	                                 d_cands.inds,
	                                 d_cands.begins,
	                                 d_cands.ends,
	                                 d_cands.filter_inds,
	                                 d_cands.dm_inds,
#ifdef SNR_BASED_CLUSTERING
                                   d_cands.peaks,
#endif
	                                 d_labels,
	                                 time_tol,
	                                 filter_tol,
	                                 dm_tol));
	// Finally, trace equivalency chains to find the final labels
	// Note: This is a parallel version of this algorithm that may not be
	//         as efficient as the sequential version but should win out
	//         in overall speed.


	unsigned int* d_counter_address;
	cudaGetSymbolAddress((void**)&d_counter_address, "d_counter");
	thrust::device_ptr<unsigned int> d_counter_ptr(d_counter_address);
	*d_counter_ptr = 0;
	
	thrust::for_each(make_counting_iterator<unsigned int>(0),
	                 make_counting_iterator<unsigned int>(count),
	                 trace_equivalency_chain<hd_size>(d_labels));
	
	//std::cout << "Total chain iterations: " << *d_counter_ptr << std::endl;
	
	// Finally we do a quick count of the number of unique labels
	//   This is efficiently achieved by checking where new labels are
	//     unchanged from their original values (i.e., where d_labels[i] == i)
	thrust::device_vector<int> d_label_roots(count);
	thrust::transform(d_labels_begin, d_labels_begin+count,
	                  make_counting_iterator<hd_size>(0),
	                  d_label_roots.begin(),
	                  thrust::equal_to<hd_size>());
	*label_count = thrust::count_if(d_label_roots.begin(),
	                                d_label_roots.end(),
	                                thrust::identity<hd_size>());
	
	return HD_NO_ERROR;
}
