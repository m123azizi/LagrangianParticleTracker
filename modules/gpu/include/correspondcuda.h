/*
* mahdi: on 6/3/2024 this was matched with barker code, the only differences are under // new ones 3wayM,2wayM and lpt::regression
* solver which is now commented, and also the one for min rpj error added by me
*/
#ifndef _CORRESPONDCUDA_H_
#define _CORRESPONDCUDA_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h> 
#include <core.hpp>
#include <correspond.hpp>

using namespace std;

namespace lpt {

	//void findMatchesHost(lpt::ImageFrameGroup& cameragroup, vector<lpt::Match::Ptr>& matches); 

	class MatchIDs {
	public:
		int ids[NUM_MATCHES];
	};

	class CameraPairCUDA {
	public:
		float F[3][3];
		int cam_a_id;
		int cam_b_id;
	};

	template < typename T >
	class KernelArray {
	public:
		KernelArray<T>(thrust::device_vector< T >& d_vec) {
			data = thrust::raw_pointer_cast(&d_vec[0]);
			size = (int)d_vec.size();
		}

		T* data;
		int size;
	};

	class PointMatcherCUDA : public lpt::Correspondence {
	public:
		typedef std::shared_ptr<lpt::PointMatcherCUDA> Ptr;
		static inline lpt::PointMatcherCUDA::Ptr create() { return std::make_shared<lpt::PointMatcherCUDA>(); }

		class Parameters {
		public:
			Parameters() : match_thresh_level(50), match_threshold(5), threshold_distance(10.0), threshold_distance_level(100) {}
			float match_threshold;
			float threshold_distance; //mahdi
			int threshold_distance_level; //mahdi
			int match_thresh_level;
		} params;

		PointMatcherCUDA();

		virtual void initialize();
		virtual void initializeEpipolarMatchThread(int thread_id);
		virtual void addControls();
		virtual void findUniqueMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches);
		virtual void findEpipolarMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap);

		// new ones 3wayM,2wayM 
		virtual void find3WayMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches);
		// a variant 3 way match without clearing matches
		virtual void find3WayMatches_mahdi(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& MatchMap, vector<lpt::Match::Ptr>& matches);
		virtual void find2WayMathces_wo_clear(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches);
		virtual void find2WayMathces(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches);

		friend void callbackMatchThreshcuda(int state, void* data) {
			PointMatcherCUDA* matcher = static_cast<PointMatcherCUDA*>(data);
			matcher->params.match_threshold = static_cast<float>(matcher->params.match_thresh_level) / 10.0;
			cout << " callbackMatchThreshcuda matcher->params.match_threshold = " << matcher->params.match_threshold  <<endl;
		}
		friend void callbackThresholdDistanceChange(int state, void* data) {
			PointMatcherCUDA* matcher = static_cast<PointMatcherCUDA*>(data);
			matcher->params.threshold_distance = static_cast<float>(matcher->params.threshold_distance_level) / 10.0;
			cout << " callbackThresholdDistanceChange matcher->params.threshold_distance = " << matcher->params.threshold_distance << endl;
		}








		// This function is used to to calculate the reprojection error
		//double reprojection_error(lpt::Match::Ptr match);
	private:
		void findEpipolarMatchesStreams(lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap);
		void findEpipolarMatchesManyThreads(lpt::ImageFrameGroup& frame_group); //TODO: try to make this work!!
		int getNextComputeDeviceID();
		thrust::host_vector<CameraPairCUDA> camera_pairs_h;
		thrust::host_vector<int> num_matches_h;
		thrust::device_vector<int> num_matches_d;
		thrust::host_vector< MatchIDs > matches2way_h;
		thrust::device_vector< MatchIDs > matches2way_d;

		MatchIDs match_initializer;

		vector<cudaStream_t> streams;

		thrust::host_vector<float> particles_x_h;
		thrust::host_vector<float> particles_y_h;

		thrust::device_vector<float> particles_x_d;
		thrust::device_vector<float> particles_y_d;

		thrust::host_vector<int> num_particles_h;
		thrust::device_vector<int> num_particles_d;

		boost::mutex mutex;
		queue<int> compute_devices_available;
		//lpt::Regression solver;
	};

} // namespace lpt

#endif // _CORRESPONDCUDA_H_
