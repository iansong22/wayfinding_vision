# Author: Ian Song
# Email: iansong2@illinois.edu
"""
Adapted from AB3DMOT
@article{Weng2020_AB3DMOT, 
	author = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris}, 
	journal = {IROS}, 
	title = {{3D Multi-Object Tracking: A Baseline and New Evaluation Metrics}}, 
	year = {2020} 
}
"""
import numpy as np, os, copy, math
from wayf_vision.kalman.box import Box3D
from wayf_vision.kalman.matching import data_association
from wayf_vision.kalman.kalman_filter import KF

HUMAN_ID = 1
OBJECT_ID = 0

np.set_printoptions(suppress=True, precision=3)

class Wayfinding_3DMOT(object):			  	
	def __init__(self, metric="giou_3d", algm = "greedy", vis_thres=0.5, lidar_thres=0.5, max_age = 5, min_hits = 1, ID_init=0, output_preds=False):             
     
		self.output_preds = output_preds
  
		# counter
		self.trackers = []
		self.frame_count = 0
		self.ID_count = [ID_init]
		self.id_now_output = []

		# debug
		# self.debug_id = 2
		self.debug_id = None

	

		# add negative due to it is the cost
		if metric in ['dist_3d', 'dist_2d', 'm_dis']: 
			vis_thres *= -1
			lidar_thres *= -1

		self.algm, self.metric, self.vis_thres, self.lidar_thres, self.max_age, self.min_hits = \
			algm, metric, vis_thres, lidar_thres, max_age, min_hits

		# define max/min values for the output affinity matrix
		if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
		elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
		elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0

	def process_dets(self, dets):
		# convert each detection into the class Box3D 
		# inputs: 
		# 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

		dets_new = []
		for det in dets:
			det_tmp = Box3D.array2bbox_raw(det)
			dets_new.append(det_tmp)

		return dets_new

	def prediction(self):
		# get predicted locations from existing tracks

		trks = []
		trks_ids = []
		for t in range(len(self.trackers)):
			
			# propagate locations
			kf_tmp = self.trackers[t]
   
			if kf_tmp.id == self.debug_id:
				print('\n before prediction')
				print(kf_tmp.kf.x.reshape((-1)))
				print('\n current velocity')
				print(kf_tmp.get_velocity())
    
			kf_tmp.kf.predict()
   
			if kf_tmp.id == self.debug_id:
				print('After prediction')
				print(kf_tmp.kf.x.reshape((-1)))
			# kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

			# update statistics
			kf_tmp.time_since_update += 1 		
			trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
			trks.append(Box3D.array2bbox(trk_tmp))
			trks_ids.append(kf_tmp.id)

		return trks, trks_ids

	def update(self, matched, lidar_matched, unmatched_trks, dets, lidar_dets):
		# update matched trackers with assigned detections
		
		dets = copy.copy(dets)
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				lidar_det = False
				d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
				# check vision detections first, then lidar detections
				if len(d) == 0:
					d = lidar_matched[np.where(lidar_matched[:, 1] == t)[0], 0]
					lidar_det = True
				assert len(d) == 1, 'error'

				# update statistics
				trk.time_since_update = 0		# reset because just updated
				trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
				if lidar_det:
					bbox3d = Box3D.bbox2array(lidar_dets[d[0]])
					# do not change class if detection is from lidar
				else:
					bbox3d = Box3D.bbox2array(dets[d[0]])
					trk.class_id = HUMAN_ID    # vision detection, set to human class
				# bbox3d = Box3D.bbox2array(dets[d[0]]) if not lidar_det else Box3D.bbox2array(lidar_dets[d[0]])
				# trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])

				if trk.id == self.debug_id:
					print('track ID %d is matched with detection ID %d' % (trk.id, d[0]))
					print('After ego-compoensation')
					print(trk.kf.x.reshape((-1)))
					print('matched measurement')
					print(bbox3d.reshape((-1)))

				# kalman filter update with observation
				trk.kf.update(bbox3d)

				if trk.id == self.debug_id:
					print('after matching')
					print(trk.kf.x.reshape((-1)))
					print('\n current velocity')
					print(trk.get_velocity())

				# trk.kf.x[3] = self.within_range(trk.kf.x[3])

			# debug use only
			# else:
				# print('track ID %d is not matched' % trk.id)

	def birth(self, dets, unmatched_dets, class_id=OBJECT_ID):
		# create and initialise new trackers for unmatched detections

		# dets = copy.copy(dets)
		new_id_list = list()					# new ID generated for unmatched detections
		for i in unmatched_dets:        			# a scalar of index
			trk = KF(Box3D.bbox2array(dets[i]), self.ID_count[0], class_id=class_id)
			self.trackers.append(trk)
			new_id_list.append(trk.id)
			# print('track ID %s has been initialized due to new detection' % trk.id)

			self.ID_count[0] += 1

		return new_id_list

	def output(self):
		# output exiting tracks that have been stably associated, i.e., >= min_hits
		# and also delete tracks that have appeared for a long time, i.e., >= max_age

		num_trks = len(self.trackers)
		results = []
		for trk in reversed(self.trackers):
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
			d = Box3D.bbox2array_raw(d)

			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits or trk.class_id == HUMAN_ID)):      
				results.append(np.concatenate((d, np.array([trk.id, trk.class_id]))).reshape(1, -1)) 		
			num_trks -= 1

			# deadth, remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(num_trks)

		return results

	def track(self, dets_all, frame=0):
		"""
		Params:
		  	dets_all: dict
				vision - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
				lidar - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
			frame:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns a dict with the following keys:
			results:  a numpy array of tracked boxes in the format [[h,w,l,x,y,z,theta,ID],...]
			affi:     a dict of affinity matrices for vision and lidar detections
			preds:    a list of predicted boxes in the format [[h,w,l,x,y,z,theta], ID]
		Note: 
			- The results are the active tracks that have been stably associated, i.e., >= min_hits.
			- The affinity matrix is the cost matrix for data association, where the values are the negative of the similarity scores.
			- The preds are the predicted boxes for each track.

		NOTE: The number of tracks returned may differ from the number of detections provided.
		"""
		dets = dets_all['vision']
		lidar_dets = dets_all['lidar']
  
		self.frame_count += 1

		# recall the last frames of outputs for computing ID correspondences during affinity processing
		self.id_past_output = copy.copy(self.id_now_output)
		self.id_past = [trk.id for trk in self.trackers]

		# process detection format
		dets = self.process_dets(dets)
		lidar_dets = self.process_dets(lidar_dets)

		# tracks propagation based on velocity
		trks, trks_ids = self.prediction()
   
		# matching
		trk_innovation_matrix = None
		if self.metric == 'm_dis':
			trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 
			
		matched, unmatched_dets_indices, unmatched_trks, cost, affi = \
			data_association(dets, trks, self.metric, self.vis_thres, self.algm, trk_innovation_matrix, trk_info=trks_ids)
		
		# matching using lidar detections

		lidar_matched, lidar_unmatched_dets, lidar_unmatched_trks, lidar_cost, lidar_affi = \
			data_association(lidar_dets, trks, self.metric, self.lidar_thres, self.algm, trk_innovation_matrix, trk_info=trks_ids)
		
		affi = {"vision" : affi, "lidar" : lidar_affi}

		# set unmatched tracks to the intersection of unmatched tracks from both vision and lidar detections
		unmatched_trks = list(set(unmatched_trks) & set(lidar_unmatched_trks))

		# print('updating with matches:')
		# for m in matched:
		# 	print(' - track ID %d (#%d) is matched with vision detection #%d' % (trks_ids[m[1]], m[1], m[0]))
		# for m in lidar_matched:
		# 	print(' - track ID %d (#%d) is matched with lidar detection #%d' % (trks_ids[m[1]], m[1], m[0]))

		# update trks with matched detection measurement
		self.update(matched, lidar_matched, unmatched_trks, dets, lidar_dets)

		# for det_idx in unmatched_dets_indices:
		# 	print('vision detection ID #%d at index %d is unmatched' % (trks_ids[det_idx], det_idx))
		# 	print('max affinity for this detection is %.3f' % np.max(affi['vision'][det_idx]))

		# for trk_idx in unmatched_trks:
			# print('lidar detection ID #%d at index %d is unmatched' % (trks_ids[trk_idx], trk_idx))
			# print('max affinity for this detection is %.3f' % np.max(affi['lidar'][:, trk_idx]))
			
		# create and initialise new trackers for unmatched detections
		new_id_list = self.birth(dets, unmatched_dets_indices, class_id=HUMAN_ID)
		new_lidar_id_list = self.birth(lidar_dets, lidar_unmatched_dets, class_id=OBJECT_ID)
		for new_id in new_id_list:
			print('track ID %d has been initialized due to new vision detection' % new_id)
		for new_id in new_lidar_id_list:
			print('track ID %d has been initialized due to new lidar detection' % new_id)

		# output existing valid tracks
		results = self.output()
		if len(results) > 0: results = np.concatenate(results)		# h,w,l,x,y,z,theta, ID
		else:            	 results = np.empty((0, 15))
		self.id_now_output = results[:, 7].tolist()					# only the active tracks that are outputed
		if self.output_preds:
			preds = []
			for i, trk in enumerate(trks):
				pred = [Box3D.bbox2array_raw(trk), trks_ids[i]]
				preds.append(pred)
			return {"results": results, "affi": affi, "preds": preds}
		return {"results": results, "affi": affi}
