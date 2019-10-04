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

template<typename T, class F>
struct merge_connected_labels : public thrust::unary_function<unsigned,bool> {
	hd_size n;
	T*      labels;
	F       connected_func;
	merge_connected_labels(hd_size n_,
	                       T* labels_,
	                       F connected_func_)
		: n(n_),
		  labels(labels_),
		  connected_func(connected_func_) {}
	inline __host__ __device__
	bool operator()(unsigned i) {
		T label0 = labels[i];
		T label  = label0;
		for( unsigned j=0; j<n; ++j ) {
			if( labels[j] < label &&
			    connected_func(i, j) ) {
				label     = labels[j];
				// Note: Updating immediately allows use by other threads
				labels[i] = label;
			}
		}
		//labels[i] = label; // Deferred update minimises memory-writes
		bool changed = (label != label0);
		return changed;
	}
};

struct check_connected_labels : public thrust::binary_function<unsigned,
                                                               unsigned,
                                                               bool> {
	const hd_size* d_samp_inds;
	const hd_size* d_begins;
	const hd_size* d_ends;
	const hd_size* d_filters;
	const hd_size* d_dms;
	hd_size   time_tol;
	hd_size filter_tol;
	hd_size     dm_tol;
	check_connected_labels(const hd_size* d_samp_inds_,
	                       const hd_size* d_begins_, const hd_size* d_ends_,
	                       const hd_size* d_filters_, const hd_size* d_dms_,
	                       hd_size time_tol_, hd_size filter_tol_, hd_size dm_tol_)
		: d_samp_inds(d_samp_inds_),
		  d_begins(d_begins_), d_ends(d_ends_),
		  d_filters(d_filters_), d_dms(d_dms_),
		  time_tol(time_tol_), filter_tol(filter_tol_), dm_tol(dm_tol_) {}
	inline __host__ __device__
	bool operator()(unsigned i, unsigned j) const {
		hd_size samp_i   = d_samp_inds[i];
		hd_size begin_i  = d_begins[i];
		hd_size end_i    = d_ends[i];
		hd_size filter_i = d_filters[i];
		hd_size dm_i     = d_dms[i];
		
		hd_size samp_j   = d_samp_inds[j];
		hd_size begin_j  = d_begins[j];
		hd_size end_j    = d_ends[j];
		hd_size filter_j = d_filters[j];
		hd_size dm_j     = d_dms[j];
		return are_coincident(samp_i, samp_j,
		                      begin_i, begin_j,
		                      end_i, end_j,
		                      filter_i, filter_j,
		                      dm_i, dm_j,
		                      time_tol, filter_tol, dm_tol);
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
	thrust::device_ptr<hd_size> d_labels_begin(d_labels);
	thrust::sequence(d_labels_begin, d_labels_begin+count);
	using thrust::make_counting_iterator;
	
	merge_connected_labels<hd_size,check_connected_labels>
		merge_functor(count,
		              d_labels,
		              check_connected_labels(d_cands.inds,
		                                     d_cands.begins,
		                                     d_cands.ends,
		                                     d_cands.filter_inds,
		                                     d_cands.dm_inds,
		                                     time_tol,
		                                     filter_tol,
		                                     dm_tol));
	// Iteratively traverse the graph of connections to update labels until convergence
	while( thrust::transform_reduce(make_counting_iterator<unsigned>(0),
	                                make_counting_iterator<unsigned>(count),
	                                merge_functor, 0, thrust::logical_or<bool>()) ) {}
	
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
