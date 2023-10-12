# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref
from libc.stdlib cimport qsort

# Import the header files
from vlsift._vl.host cimport *
from vlsift._vl.sift cimport *
from vlsift.cy_util cimport set_python_vl_printf


cdef int korder(const void *a, const void *b) noexcept nogil:
    cdef float x = (<float*> a)[2] - (<float*> b)[2]
    if x < 0: return -1
    if x > 0: return +1
    return 0


@cython.boundscheck(False)
cpdef cy_sift(float[:, ::1] data, int n_octaves,
              int n_levels, int first_octave, float peak_threshold,
              float edge_threshold, float norm_threshold, int magnification,
              int window_size, float[:, ::1] frames, bint force_orientations,
              bint float_descriptors, bint compute_descriptor, bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        bint is_first_octave = True
        int n_user_keypoints = 0, total_keypoints = 0, reserved = 0, i = 0, j = 0, q = 0
        int height = data.shape[0]
        int width = data.shape[1]
        float *user_keypoints_arr
        VlSiftFilt *filt = vl_sift_new(width, height, n_octaves, n_levels,
                                       first_octave)

        # Create empty 2D output arrays
        float[:, ::1] out_descriptors = np.empty((0, 128), dtype=np.float32,
                                                 order='C')
        float[:, ::1] out_frames = np.empty((0, 4), dtype=np.float32, order='C')

        float *flat_descriptors = &out_descriptors[0, 0]
        float *flat_out_frames = &out_frames[0, 0]

        int is_octaves_complete = 0
        const VlSiftKeypoint *keypoints
        int n_keypoints = 0

        double[4] angles
        int n_angles = 0
        const VlSiftKeypoint *curr_keypoint
        VlSiftKeypoint ik

        vl_sift_pix[128] single_descriptor_arr

        bint user_specified_frames = False

    user_specified_frames = frames is not None
    if user_specified_frames:
        n_user_keypoints = frames.shape[0]
        user_keypoints_arr = &frames[0, 0]
        # Ensure frames array is sorted by increasing scale
        qsort(user_keypoints_arr, n_user_keypoints,4 * sizeof(float), korder)

    if peak_threshold  >= 0: vl_sift_set_peak_thresh(filt, peak_threshold)
    if edge_threshold  >= 0: vl_sift_set_edge_thresh(filt, edge_threshold)
    if norm_threshold  >= 0: vl_sift_set_norm_thresh(filt, norm_threshold)
    if magnification   >= 0: vl_sift_set_magnif(filt, magnification)
    if window_size     >= 0: vl_sift_set_window_size(filt, window_size)

    if verbose:
        print("vl_sift: filter settings:")
        print("vl_sift:   octaves      (O)      = %d" % (vl_sift_get_noctaves(filt)))
        print("vl_sift:   levels       (S)      = %d" % (vl_sift_get_nlevels(filt)))
        print("vl_sift:   first octave (o_min)  = %d" % (vl_sift_get_octave_first(filt)))
        print("vl_sift:   edge thresh           = %g" % (vl_sift_get_edge_thresh(filt)))
        print("vl_sift:   peak thresh           = %g" % (vl_sift_get_peak_thresh(filt)))
        print("vl_sift:   norm thresh           = %g" % (vl_sift_get_norm_thresh(filt)))
        print("vl_sift:   window size           = %g" % (vl_sift_get_window_size(filt)))
        print("vl_sift:   magnification         = %g" % (vl_sift_get_magnif(filt)))
        print("vl_sift: return descriptor as float"
              if float_descriptors
              else "vl_sift: return descriptor as uint8")
        print("vl_sift: will source frames? yes (%d read)" % n_user_keypoints
              if user_specified_frames
              else "vl_sift: will source frames? no")
        print("vl_sift: will force orientations? %d" % (force_orientations))


    if user_specified_frames:
        # If we have specified the frames, we know how many frames
        # will be calculated, so we can skip the dynamic reallocation
        # which is normally done inside the keypoints loop
        out_frames = np.resize(out_frames, (n_user_keypoints, 4))
        flat_out_frames = &out_frames[0, 0]
        # Similar for the descriptors, if necessary
        if compute_descriptor:
            out_descriptors = np.resize(out_descriptors,
                                        (n_user_keypoints, 128))
            flat_descriptors = &out_descriptors[0, 0]

    # Process each octave
    while True:
        if verbose:
            print("vl_sift: processing octave %d" %
                      vl_sift_get_octave_index(filt))

        # Calculate the GSS for the next octave ....................
        if is_first_octave:
            is_octaves_complete = vl_sift_process_first_octave(filt,
                                                               &data[0, 0])
            is_first_octave = False
        else:
            is_octaves_complete = vl_sift_process_next_octave(filt)

        if is_octaves_complete:
            break

        if verbose:
            print("vl_sift: GSS octave %d computed" %
                  vl_sift_get_octave_index(filt))

        # Run detector .............................................
        if not user_specified_frames:
            i = 0
            vl_sift_detect(filt)

            keypoints = vl_sift_get_keypoints(filt)
            n_keypoints = vl_sift_get_nkeypoints(filt)

            if verbose:
                print("vl_sift: detected %d (unoriented) keypoints" % n_keypoints)
        else:
            n_keypoints = n_user_keypoints

        # For each keypoint
        while i < n_keypoints:
            # Obtain keypoint orientations
            if user_specified_frames:
                vl_sift_keypoint_init(filt, &ik,
                                      user_keypoints_arr[4 * i + 1],
                                      user_keypoints_arr[4 * i + 0],
                                      user_keypoints_arr[4 * i + 2])

                if ik.o != vl_sift_get_octave_index(filt):
                    break

                curr_keypoint = &ik

                # Optionally force computation of orientations
                if force_orientations:
                    n_angles = vl_sift_calc_keypoint_orientations(filt, angles,
                                                                  curr_keypoint)
                else:
                    angles[0] = user_keypoints_arr[4 * i + 3]
                    n_angles  = 1
            else:
                # This is equivalent to &keypoints[i] - just get the pointer
                # to the i'th element.
                curr_keypoint = keypoints + i
                n_angles = vl_sift_calc_keypoint_orientations(filt, angles,
                                                              curr_keypoint)

            # For each orientation
            for q in range(n_angles):
                if compute_descriptor:
                    vl_sift_calc_keypoint_descriptor(filt,
                                                     single_descriptor_arr,
                                                     curr_keypoint, angles[q])

                # Dynamically reallocate the output arrays so that they can
                # fit all the keypoints being requested.
                # If statement says: IF we will run out of space next iteration
                #                    AND we have computed the frame OR the user
                #                        has allowed estimation of the number of
                #                        orientations AND there was more than one
                #                    THEN reallocate memory
                if (reserved < total_keypoints + 1 and
                   (not user_specified_frames or
                    (force_orientations and n_angles > 1))):
                    reserved += 2 * n_keypoints

                    out_frames = np.resize(out_frames, (reserved, 4))
                    flat_out_frames = &out_frames[0, 0]

                    if compute_descriptor:
                        out_descriptors = np.resize(out_descriptors,
                                                    (reserved, 128))
                        flat_descriptors = &out_descriptors[0, 0]

                # Notice that this method will give different results
                # from MATLAB because MATLAB actually runs on the
                # transpose of the image due to it's fortran ordering!
                flat_out_frames[total_keypoints * 4 + 0] = curr_keypoint.y
                flat_out_frames[total_keypoints * 4 + 1] = curr_keypoint.x
                flat_out_frames[total_keypoints * 4 + 2] = curr_keypoint.sigma
                flat_out_frames[total_keypoints * 4 + 3] = angles[q]

                if compute_descriptor:
                    for j in range(128):
                        flat_descriptors[total_keypoints * 128 + j] = \
                            min(512.0 * single_descriptor_arr[j], 255.0)

                total_keypoints += 1
            i += 1

    if verbose:
        print("vl_sift: found %d keypoints" % (total_keypoints))

    # cleanup
    vl_sift_delete(filt)

    # If we have dynamically allocated memory for the frames, make sure that
    # we resize the array back to the correct size (since we optimistically
    # allocated previously to reduce the number of total resizes)
    if out_frames.shape[0] != total_keypoints:
        out_frames = np.resize(out_frames, (total_keypoints, 4))
        out_descriptors = np.resize(out_descriptors, (total_keypoints, 128))

    if compute_descriptor:
        if float_descriptors:
            return np.asarray(out_frames), np.asarray(out_descriptors)
        else:
            return np.asarray(out_frames), np.asarray(out_descriptors).astype(np.uint8)
    else:
        return np.asarray(out_frames)
