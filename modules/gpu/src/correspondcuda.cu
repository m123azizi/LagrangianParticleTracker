#include <correspondcuda.h>
// mahdi: on 6/3/2024 this was checked with douglas code visually and is the same.
// initiliaze epipolar match thread has some differnces, in wehre resizing for the number of streams

using namespace std;

__constant__ lpt::CameraPairCUDA pairs_k[100];

__constant__ int num_matches_k[1];

//__constant__ float pca_data_constant[300];

//__constant__ float x_mean[30];

//__constant__ float x_std[30];

//__constant__ float lr_data_constant[11];


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code == cudaErrorLaunchTimeout) {
		fprintf(stderr, "Timeout Error: The kernel was terminated due to execution time limit (watchdog timer), %s %d\n", file, line);
		if (abort) exit(code);
	}
	else if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void calcEpipolarResidualAllInOneStreams_kernel(int p, float match_threshold, lpt::KernelArray<float> particles_x, lpt::KernelArray<float> particles_y, lpt::KernelArray<int> num_particles, lpt::KernelArray<lpt::MatchIDs> matches2way, lpt::KernelArray<int> num_matches)
{
	p += blockIdx.x;
	int b = blockIdx.y * blockDim.x + threadIdx.x;
	int id_b = b;
	if (pairs_k[p].cam_b_id != 0)
		id_b += num_particles.data[pairs_k[p].cam_b_id - 1];

	if (id_b < num_particles.data[pairs_k[p].cam_b_id]) {
		float line[3];
		float x = particles_x.data[id_b];
		float y = particles_y.data[id_b];

		line[0] = pairs_k[p].F[0][0] * x + pairs_k[p].F[1][0] * y + pairs_k[p].F[2][0] * 1.f;
		line[1] = pairs_k[p].F[0][1] * x + pairs_k[p].F[1][1] * y + pairs_k[p].F[2][1] * 1.f;
		line[2] = pairs_k[p].F[0][2] * x + pairs_k[p].F[1][2] * y + pairs_k[p].F[2][2] * 1.f;
		float factor = line[0] * line[0] + line[1] * line[1];
		factor = factor ? 1.f / sqrtf(factor) : 1.f;
		line[0] *= factor;
		line[1] *= factor;
		line[2] *= factor;

		int id_a_start = 0;
		if (pairs_k[p].cam_a_id != 0)
			id_a_start = num_particles.data[pairs_k[p].cam_a_id - 1];
		int id_a_end = num_particles.data[pairs_k[p].cam_a_id];

		int match_id = b;
		if (p != 0)
			match_id += num_matches.data[p - 1];
		int match_count = 0;
		for (int id_a = id_a_start; id_a < id_a_end; ++id_a) {
			bool matched = static_cast<int>(floor(match_threshold / fabs(particles_x.data[id_a] * line[0] + particles_y.data[id_a] * line[1] + line[2])));
			if (matched && match_count < num_matches_k[0]) {
				matches2way.data[match_id].ids[match_count] = id_a;
				++match_count;
			}
		}
	}
}

__global__ void calcEpipolarResidualAllInOne_kernel(float match_threshold, lpt::KernelArray<float> particles_x, lpt::KernelArray<float> particles_y, lpt::KernelArray<int> num_particles, lpt::KernelArray<lpt::MatchIDs> matches2way, lpt::KernelArray<int> num_matches)
{
	//printf(" mahdi this is under calcEpipolarResidualAllInOne_kernel in cuda\n");//mahdi to make sure this is called
	int p = blockIdx.x;
	int b = blockIdx.y * blockDim.x + threadIdx.x;
	int id_b = b;
	if (pairs_k[p].cam_b_id != 0)
		id_b = num_particles.data[pairs_k[p].cam_b_id - 1] + b;

	if (id_b < num_particles.data[pairs_k[p].cam_b_id]) {
		float line[3];
		float x = particles_x.data[id_b];
		float y = particles_y.data[id_b];

		line[0] = pairs_k[p].F[0][0] * x + pairs_k[p].F[1][0] * y + pairs_k[p].F[2][0] * 1.f;
		line[1] = pairs_k[p].F[0][1] * x + pairs_k[p].F[1][1] * y + pairs_k[p].F[2][1] * 1.f;
		line[2] = pairs_k[p].F[0][2] * x + pairs_k[p].F[1][2] * y + pairs_k[p].F[2][2] * 1.f;
		float factor = line[0] * line[0] + line[1] * line[1];
		factor = factor ? 1.f / sqrtf(factor) : 1.f;
		line[0] *= factor;
		line[1] *= factor;
		line[2] *= factor;

		int id_a_start = 0;
		if (pairs_k[p].cam_a_id != 0)
			id_a_start = num_particles.data[pairs_k[p].cam_a_id - 1];
		int id_a_end = num_particles.data[pairs_k[p].cam_a_id];
		int match_id = b;
		if (p != 0)
			match_id += num_matches.data[p - 1];
		int match_count = 0;
		for (int id_a = id_a_start; id_a < id_a_end; ++id_a) {
			bool matched = static_cast<int>(floor(match_threshold / fabs(particles_x.data[id_a] * line[0] + particles_y.data[id_a] * line[1] + line[2])));
			if (matched && match_count < num_matches_k[0]) {
				matches2way.data[match_id].ids[match_count] = id_a;
				++match_count;
			}
		}
	}
}

__global__ void calcEpipolarLines_kernel(lpt::KernelArray<float> particles_x, lpt::KernelArray<float> particles_y, lpt::KernelArray<int> num_particles, lpt::KernelArray<float> lines_x, lpt::KernelArray<float> lines_y, lpt::KernelArray<float> lines_z, lpt::KernelArray<int> num_lines)
{
	//cout << " mahdi this is under calcEpipolarLines_kernel in cuda" << endl;
	int p = blockIdx.x;
	int b = blockIdx.y * blockDim.x + threadIdx.x;
	int cam_b_id = pairs_k[p].cam_b_id;
	int id_b = b;
	if (cam_b_id != 0)
		id_b = num_particles.data[cam_b_id - 1] + b;

	if (id_b < num_particles.data[cam_b_id]) {

		float x = particles_x.data[id_b];
		float y = particles_y.data[id_b];
		int line_id = b;
		if (p != 0)
			line_id += num_lines.data[p - 1];

		lines_x.data[line_id] = pairs_k[p].F[0][0] * x + pairs_k[p].F[1][0] * y + pairs_k[p].F[2][0] * 1.f;
		lines_y.data[line_id] = pairs_k[p].F[0][1] * x + pairs_k[p].F[1][1] * y + pairs_k[p].F[2][1] * 1.f;
		lines_z.data[line_id] = pairs_k[p].F[0][2] * x + pairs_k[p].F[1][2] * y + pairs_k[p].F[2][2] * 1.f;
		float factor = lines_x.data[line_id] * lines_x.data[line_id] + lines_y.data[line_id] * lines_y.data[line_id];
		factor = factor ? 1.f / sqrtf(factor) : 1.f;
		lines_x.data[line_id] *= factor;
		lines_y.data[line_id] *= factor;
		lines_z.data[line_id] *= factor;
	}
}

__global__ void calcEpipolarResiduals_kernel(float match_threshold, lpt::KernelArray<float> particles_x, lpt::KernelArray<float> particles_y, lpt::KernelArray<int> num_particles, lpt::KernelArray<float> lines_x, lpt::KernelArray<float> lines_y, lpt::KernelArray<float> lines_z, lpt::KernelArray<int> num_lines, lpt::KernelArray<lpt::MatchIDs> matches2way, lpt::KernelArray<int> num_matches)
{
	//cout << " mahdi this is under calcEpipolarResiduals_kernel in cuda" << endl;
	////__shared__ float line[3];
	//
	//int line_id = blockIdx.x;
	////float line[] = {lines_x.data[line_id], lines_y.data[line_id], lines_z.data[line_id]};
	//int cam_id;
	//int r_id = residuals.size;
	//for (int pair_id = 0; pair_id < num_lines.size; ++pair_id) {
	//	if( line_id < num_lines.data[pair_id] ) {
	//		cam_id = pairs_k[pair_id].cam_a_id;
	//		if (pair_id !=0 )
	//			r_id = num_residuals.data[pair_id - 1] + (line_id - num_lines.data[pair_id -1]) * (num_particles.data[cam_id] - num_particles.data[cam_id - 1]) + threadIdx.x;
	//		else
	//			r_id = (line_id - num_lines.data[pair_id -1]) * (num_particles.data[cam_id] - num_particles.data[cam_id - 1]) + threadIdx.x;
	//		break;
	//	}
	//}

	////if (threadIdx.x == 0) {
	////	line[0] = lines_x.data[line_id];
	////	line[1] = lines_y.data[line_id];
	////	line[2] = lines_z.data[line_id];
	////}	
	////__syncthreads(); //FIXME: put lines in constant memory if possible

	//int id_a = num_particles.data[cam_id - 1] + blockIdx.y * blockDim.x + threadIdx.x;
	//
	//if ( id_a < num_particles.data[cam_id] && r_id < residuals.size ) {
	//	residuals.data[r_id] = static_cast<int>( match_threshold / fabs( particles_x.data[id_a] * lines_x.data[line_id]  + particles_y.data[id_a] * lines_y.data[line_id] + lines_z.data[line_id] ) );
	//}

}

//__global__ void test_kernel_corresponde(float* sample, float* prob) {
//	__shared__ float temp_ten[10];
//
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (i < 30) {
//		sample[i] = (sample[i] - x_mean[i]) / x_std[i];
//
//	}
//
//	__syncthreads();
//
//	//if (i < 30) {
//	//	for (int j = 0; j < 10; j++) {
//	//		atomicAdd(&temp_ten[j], sample[i] * pca_data_constant[i * 10 + j]);
//	//	}
//	//}
//
//	__syncthreads();
//
//	if (i < 10) {
//
//		temp_ten[i] = temp_ten[i] * lr_data_constant[i];
//
//	}
//
//	__syncthreads();
//
//	if (i == 0) {
//		for (int j = 0; j < 10; j++) {
//			prob[0] += temp_ten[j];
//
//		}
//
//		prob[0] += lr_data_constant[10];
//		prob[0] = 1 / (1 + exp(-prob[0]));
//
//	}
//
//
//}

namespace lpt {

	PointMatcherCUDA::PointMatcherCUDA() {
		cout << "Epipolor Point matcher created (CUDA Enabled)" << endl;
		int devcount = 0;
		cudaGetDeviceCount(&devcount);
		for (int i = 0; i < devcount; ++i) {
			cudaDeviceProp device_prop;
			cudaGetDeviceProperties(&device_prop, i);
			if (!device_prop.kernelExecTimeoutEnabled) {
				cout << "Device " << i << ":  " << device_prop.name << "  added to available queue" << endl;
			}
			else {
				cout << "Device " << i << ":  " << device_prop.name << " added to available queue (Kernel run time limited)" << endl;
			}
			compute_devices_available.push(i);
		}
	}

	int PointMatcherCUDA::getNextComputeDeviceID() {
		boost::mutex::scoped_lock(this->mutex);
		int id = this->compute_devices_available.front();
		this->compute_devices_available.pop();
		return id;
	}

	void PointMatcherCUDA::initializeEpipolarMatchThread(int thread_id) {
		auto& cameras = shared_objects->cameras;
		auto& camera_pairs = shared_objects->camera_pairs;
		int id = getNextComputeDeviceID();
		cudaSetDevice(id);
		cout << "PointMatcherCUDA Thread " << thread_id << " setting device " << id << endl;

		particles_x_h.resize(this->initial_max_particles_per_image * cameras.size(), 0.f);
		particles_x_d = particles_x_h;

		particles_y_h.resize(this->initial_max_particles_per_image * cameras.size(), 0.f);
		particles_y_d = particles_y_h;

		num_particles_h.resize(cameras.size(), 0);
		num_particles_d = num_particles_h;

		camera_pairs_h.resize(camera_pairs.size());

		num_matches_h.resize(camera_pairs.size(), 0);
		num_matches_d = num_matches_h;

		for (int i = 0; i < NUM_MATCHES; ++i)
			this->match_initializer.ids[i] = -1;

		matches2way_h.resize(camera_pairs.size() * this->initial_max_particles_per_image, this->match_initializer);
		matches2way_d = matches2way_h;

		for (int i = 0; i < camera_pairs.size(); ++i) {
			for (int n = 0; n < 3; ++n)
				for (int m = 0; m < 3; ++m)
					camera_pairs_h[i].F[n][m] = static_cast<float>(camera_pairs[i].F[n][m]);
			camera_pairs_h[i].cam_a_id = camera_pairs[i].cam_A.id;
			camera_pairs_h[i].cam_b_id = camera_pairs[i].cam_B.id;
		}
		cout << "camera_pairs_h.size() " << camera_pairs_h.size() << endl; // 28 correct
		streams.clear();
		for (int f = 2; f < camera_pairs_h.size(); ++f) {
			if (camera_pairs_h.size() % f == 0) {
				streams.resize(f);
				break;
			}
		}
		cout << "Streams size = " << streams.size() << endl;
		for (int i = 0; i < streams.size(); ++i)
			cudaStreamCreate(&(streams[i]));

		int num = NUM_MATCHES;

		gpuErrchk(cudaMemcpyToSymbol(num_matches_k, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpyToSymbol(pairs_k, thrust::raw_pointer_cast(camera_pairs_h.data()), sizeof(CameraPairCUDA) * camera_pairs_h.size()));

		//mahdi two lines below have similar functioning to barker but didnt'w work, later check why, shows zero 3d particles
		/*cudaMemcpyToSymbol("num_matches_k", &num, sizeof(int));
		cudaMemcpyToSymbol("pairs_k", thrust::raw_pointer_cast(&camera_pairs_h[0]), sizeof(CameraPairCUDA) * camera_pairs_h.size());*/

		cout << "Succeed to copy to constant memory ******" << endl;
		//gpuErrchk(cudaMemcpyToSymbol(x_mean, x_train_mean.data(), sizeof(float) * 30, 0, cudaMemcpyHostToDevice));
		//gpuErrchk(cudaMemcpyToSymbol(x_std, x_train_std.data(), sizeof(float) * 30, 0, cudaMemcpyHostToDevice));
		//gpuErrchk(cudaMemcpyToSymbol(lr_data_constant, lr_data.data(), sizeof(float) * 11, 0, cudaMemcpyHostToDevice));

		/*vector<float> pca_data_flatten;

		for (const auto &v : pca_parameters) {
			pca_data_flatten.insert(pca_data_flatten.end(), v.begin(), v.end());
		}

		gpuErrchk(cudaMemcpyToSymbol(pca_data_constant, pca_data_flatten.data(), sizeof(float) * 300, 0, cudaMemcpyHostToDevice));*/






		//float* temp_device = NULL;
		//vector<float> test_ones(30, 1.0f);

		//gpuErrchk(cudaMalloc((void**)(&temp_device), 30 * sizeof(float)));
		//gpuErrchk(cudaMemcpy(temp_device, test_ones.data(), sizeof(float) * 30, cudaMemcpyHostToDevice));

		//float prob[1];
		//prob[0] = 0.0f;

		//float* prob_device = NULL;
		//gpuErrchk(cudaMalloc((void**)(&prob_device), 1 * sizeof(float)));
		//gpuErrchk(cudaMemcpy(prob_device, prob, sizeof(float) * 1, cudaMemcpyHostToDevice));

		//dim3 dimBlock(32, 1, 1);
		//dim3 dimGrid(1, 1, 1);
		//test_kernel_corresponde << <dimGrid, dimBlock, 0 >> > (temp_device, prob_device);

		//gpuErrchk(cudaMemcpy(prob, prob_device, 1 * sizeof(float), cudaMemcpyDeviceToHost));
		//gpuErrchk(cudaDeviceSynchronize());
		//gpuErrchk(cudaPeekAtLastError());

		//cout << "prob is: " << prob[0] << endl;
	}

	void PointMatcherCUDA::initialize() {
		this->initializeMatchMap();
	}

	void PointMatcherCUDA::addControls() {
		void* matcher_void_ptr = static_cast<void*> (this);
		cv::createTrackbar("Match Threshcuda", string(), &params.match_thresh_level, 500, callbackMatchThreshcuda, matcher_void_ptr);
		cv::createTrackbar("Threshold Distance for min erj err", string(), &params.threshold_distance_level, 200, callbackThresholdDistanceChange, matcher_void_ptr);//mahdi: this defines the distance threshold for the function returning minimum match and reprojection error
	}

	void PointMatcherCUDA::findEpipolarMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap) {

		thrust::fill(matches2way_d.begin(), matches2way_d.end(), match_initializer);
		//cout << " mahdi this is under findfpipolarmatches under correspond in cuda" << endl;
		num_particles_h[0] = frame_group[0].particles.size();
		for (int p = 0; p < frame_group[0].particles.size(); ++p) {
			particles_x_h[p] = static_cast<float>(frame_group[0].particles[p]->x);
			particles_y_h[p] = static_cast<float>(frame_group[0].particles[p]->y);
		}

		int max_particles = num_particles_h[0];
		for (int i = 1; i < frame_group.size(); ++i) {
			num_particles_h[i] = frame_group[i].particles.size() + num_particles_h[i - 1];
			for (int p = 0; p < frame_group[i].particles.size(); ++p) {
				particles_x_h[num_particles_h[i - 1] + p] = static_cast<float>(frame_group[i].particles[p]->x);
				particles_y_h[num_particles_h[i - 1] + p] = static_cast<float>(frame_group[i].particles[p]->y);
			}
			if (frame_group[i].particles.size() > max_particles) {
				max_particles = frame_group[i].particles.size();
				if (max_particles > this->initial_max_particles_per_image)
					cout << "WARNING IMAGE FRAME HAS EXCEEDED MAX PARTICLES:  correspondcuda.cu" << endl;
			}
		}

		cudaMemcpyAsync(thrust::raw_pointer_cast(&num_particles_d[0]), thrust::raw_pointer_cast(&num_particles_h[0]), num_particles_h.size() * sizeof(int), cudaMemcpyHostToDevice, streams[0]);
		gpuErrchk(cudaGetLastError());
		cudaMemcpyAsync(thrust::raw_pointer_cast(&particles_x_d[0]), thrust::raw_pointer_cast(&particles_x_h[0]), *num_particles_h.rbegin() * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
		gpuErrchk(cudaGetLastError());
		cudaMemcpyAsync(thrust::raw_pointer_cast(&particles_y_d[0]), thrust::raw_pointer_cast(&particles_y_h[0]), *num_particles_h.rbegin() * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
		gpuErrchk(cudaGetLastError());

		num_matches_h[0] = frame_group[camera_pairs_h[0].cam_b_id].particles.size();
		for (int i = 1; i < this->camera_pairs_h.size(); ++i) {
			num_matches_h[i] = frame_group[camera_pairs_h[i].cam_b_id].particles.size() + num_matches_h[i - 1];
		}

		cudaMemcpyAsync(thrust::raw_pointer_cast(&num_matches_d[0]), thrust::raw_pointer_cast(&num_matches_h[0]), num_matches_h.size() * sizeof(int), cudaMemcpyHostToDevice, streams[0]);
		gpuErrchk(cudaGetLastError());

		int num_pairs = camera_pairs_h.size();

		dim3 dimblock(256, 1, 1); // mahdi was 128
		dim3 dimgrid(static_cast<unsigned int>(num_pairs), (static_cast<unsigned int>(max_particles) / dimblock.x) + 1);

		calcEpipolarResidualAllInOne_kernel << < dimgrid, dimblock, 0, streams[0] >> > (params.match_threshold, particles_x_d, particles_y_d, num_particles_d, matches2way_d, num_matches_d);
		gpuErrchk(cudaGetLastError());

		//cout << " 0000000000 " << endl;
		cudaMemcpyAsync(thrust::raw_pointer_cast(&matches2way_h[0]), thrust::raw_pointer_cast(&matches2way_d[0]), *num_matches_h.rbegin() * sizeof(MatchIDs), cudaMemcpyDeviceToHost, streams[0]);
		gpuErrchk(cudaGetLastError());

		cudaStreamSynchronize(streams[0]);
		gpuErrchk(cudaGetLastError());

		int match_overload = 0;
		for (int p = 0; p < camera_pairs_h.size(); ++p) {
			int match_id = (p == 0 ? 0 : num_matches_h[p - 1]);
			int cam_b = camera_pairs_h[p].cam_b_id;
			int cam_a = camera_pairs_h[p].cam_a_id;
			int b_end = num_particles_h[cam_b];
			int b_start = (cam_b != 0 ? num_particles_h[cam_b - 1] : 0);
			int a_start = (cam_a != 0 ? num_particles_h[cam_a - 1] : 0);
			for (int b_id = b_start; b_id < b_end; ++b_id, ++match_id) {
				for (int m = 0; m < NUM_MATCHES; ++m) {
					int a_id = matches2way_h[match_id].ids[m];
					if (a_id >= 0) {
						//matchmap[b_id][cam_a][m] = a_id - a_start;
						//matchmap[a_id][cam_b][m] = b_id - b_start; 
						auto itb = std::find(matchmap[b_id][cam_a].begin(), matchmap[b_id][cam_a].end(), -1);
						auto ita = std::find(matchmap[a_id][cam_b].begin(), matchmap[a_id][cam_b].end(), -1);
						if (itb != matchmap[b_id][cam_a].end() && ita != matchmap[a_id][cam_b].end()) {
							*itb = a_id - static_cast<int>(a_start);
							*ita = static_cast<int>(b_id - b_start);
						}
						else
							match_overload++;
					}
					else
						break;
				}
			}
		}
		if (match_overload > 0)
			;// cout << "WARNING: MORE MATCHES THAN ARRAY SIZE NUM_MATCHES: total overload = " << match_overload << endl;
	}
	//
	//// This function is used to to calculate the reprojection error
	//double PointMatcherCUDA::reprojection_error(lpt::Match::Ptr match) {
	//	auto& cameras = shared_objects->cameras;
	//
	//	auto& S = this->shared_objects->S;
	//	auto& P = this->shared_objects->P;
	//	double temp_x, temp_y, temp_z;
	//
	//	vector<cv::Point3d> object_points(1);
	//	vector<cv::Point2d> image_points(1);
	//
	//	vector<vector<cv::Point2d>> total_image_points(match->particles.size(), image_points);
	//
	//	size_t number_of_cams = match->particles.size();
	//	if (number_of_cams >= 2) {
	//		double empty[3] = { 0 };
	//		double empty1 = 0;
	//		double** A = new double*[2 * number_of_cams];
	//		double* B = new double[2 * number_of_cams];
	//		double* X = new double[3];
	//
	//		int id = -1;
	//		for (int i = 0; i < number_of_cams; ++i) {
	//			int s = i * 2;
	//			int e = s + 1;
	//			A[s] = new double[3];
	//			A[e] = new double[3];
	//			lpt::ParticleImage::Ptr P_img = match->particles[i].first;
	//			size_t camID = match->particles[i].second;
	//			double x = (P_img->x - cameras[camID].c[0]) / (1. * cameras[camID].f[0]);    // FIXME: May need -1 multiplier for pixel coordinate system (upper left corner origin)
	//			double y = (P_img->y - cameras[camID].c[1]) / (1. * cameras[camID].f[1]);    // Convert P.x and P.y to normalized coordinates through intrinsic parameters
	//			id = P_img->id;
	//			A[s][0] = x * cameras[camID].R[2][0] - cameras[camID].R[0][0];
	//			A[s][1] = x * cameras[camID].R[2][1] - cameras[camID].R[0][1];
	//			A[s][2] = x * cameras[camID].R[2][2] - cameras[camID].R[0][2];
	//			A[e][0] = y * cameras[camID].R[2][0] - cameras[camID].R[1][0];
	//			A[e][1] = y * cameras[camID].R[2][1] - cameras[camID].R[1][1];
	//			A[e][2] = y * cameras[camID].R[2][2] - cameras[camID].R[1][2];
	//
	//			B[s] = cameras[camID].T[0] - x * cameras[camID].T[2];
	//			B[e] = cameras[camID].T[1] - y * cameras[camID].T[2];
	//		}
	//
	//		solver.Householder(A, B, number_of_cams * 2, 3);   //Transform A into upper triangular form
	//		solver.Backsub(A, B, 3, X);
	//		//array<lpt::Particle3d::float_type, lpt::Particle3d::dim> coords = {{X[0], X[1], X[2]}};
	//
	//		cv::Point3d p(X[0], X[1], X[2]);
	//
	//		object_points[0] = p;
	//
	//		for (int i = 0; i < match->particles.size(); ++i) {
	//			size_t camID = match->particles[i].second;
	//
	//			cv::Mat R = cv::Mat(3, 3, CV_64F, cameras[camID].R);
	//			cv::Mat t_vec = cv::Mat(3, 1, CV_64F, cameras[camID].T);
	//			cv::Mat r_vec = cv::Mat::zeros(3, 1, CV_64F);
	//			cv::Rodrigues(R, r_vec);
	//
	//			cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[camID].getCameraMatrix(), cameras[camID].getDistCoeffs(), total_image_points[i]);
	//		}
	//
	//		// calculate the reprojection error
	//		double error = 0.0;
	//		double temp_error = 0.0;
	//		double match_x, match_y, repo_x, repo_y;
	//
	//		for (int i = 0; i < match->particles.size(); i++) {
	//			match_x = match->particles[i].first->x;
	//			match_y = match->particles[i].first->y;
	//
	//			repo_x = total_image_points[i][0].x;
	//			repo_y = total_image_points[i][0].y;
	//
	//			temp_error = sqrt((match_x - repo_x) * (match_x - repo_x) + (match_y - repo_y) * (match_y - repo_y));
	//			error += temp_error;
	//		}
	//
	//		for (int i = 0; i < number_of_cams * 2; ++i)
	//			delete[] A[i];
	//		delete[] A;
	//		delete[] B;
	//		delete[] X;
	//
	//		error = error / (double)match->particles.size();
	//		return error;
	//	}
	//}

	// mahdi below original from barker
	void PointMatcherCUDA::findUniqueMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
		vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched)
					for (int cam_b = cam_a + 1; cam_b < frame_group.size() - 1; ++cam_b) {
						int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
						for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
							int b = matchmap[a + a_start][cam_b][match_ab];
							if (b < 0)
								break;
							lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];

							if (!Pb->is_4way_matched)
								for (int cam_c = cam_b + 1; cam_c < frame_group.size(); ++cam_c) {
									int c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
									for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc) {
										int c = matchmap[b + b_start][cam_c][match_bc];
										if (c < 0)
											break;

										lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];

										if (!Pc->is_4way_matched && std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c))
											for (int cam_d = cam_c + 1; cam_d < frame_group.size(); ++cam_d) {
												vector<lpt::Match::Ptr> matches4way;
												int d_start = (cam_d != 0 ? num_particles[cam_d - 1] : 0);
												for (int match_cd = 0; match_cd < NUM_MATCHES; ++match_cd) {
													int d = matchmap[c + c_start][cam_d][match_cd];
													if (d < 0)
														break;
													lpt::ParticleImage::Ptr Pd = frame_group[cam_d].particles[d];
													if (!Pd->is_4way_matched && std::count(matchmap[a + a_start][cam_d].begin(), matchmap[a + a_start][cam_d].end(), d) && std::count(matchmap[b + b_start][cam_d].begin(), matchmap[b + b_start][cam_d].end(), d)) {
														if (!Pa->is_4way_matched && !Pb->is_4way_matched && !Pc->is_4way_matched && !Pd->is_4way_matched) {
															lpt::Match::Ptr newmatch = lpt::Match::create();
															newmatch->addParticle(Pa, cam_a);
															newmatch->addParticle(Pb, cam_b);
															newmatch->addParticle(Pc, cam_c);
															newmatch->addParticle(Pd, cam_d);
															matches4way.push_back(std::move(newmatch));
															Pa->is_4way_matched = true;
															Pb->is_4way_matched = true;
															Pc->is_4way_matched = true;
															Pd->is_4way_matched = true;
															match_ab = NUM_MATCHES;
															match_bc = NUM_MATCHES;
															match_cd = NUM_MATCHES;

														}
													}
												}
												std::move(matches4way.begin(), matches4way.end(), std::back_inserter(matches));
											}
									}
								}
						}
					}
			}
		}
	}

	////mahdi below with min rpj error
	////----------------
	//void PointMatcherCUDA::findUniqueMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {

	//	vector <MatchAndRpjErrs> matchAndErr;
	//	vector <MatchAndRpjErrs> minErrMatches;

	//	vector<int> num_particles(frame_group.size());
	//	num_particles[0] = frame_group[0].particles.size();

	//	for (int i = 1; i < frame_group.size(); ++i)
	//		num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

	//	for (int cam_a = 0; cam_a < frame_group.size() - 3; ++cam_a) {
	//		int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
	//		for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
	//			lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
	//			for (int cam_b = cam_a + 1; cam_b < frame_group.size() - 2; ++cam_b) {
	//				int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
	//				for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
	//					int b = matchmap[a + a_start][cam_b][match_ab];
	//					if (b < 0) {
	//						break;
	//					}
	//					lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];
	//					for (int cam_c = cam_b + 1; cam_c < frame_group.size() - 1; ++cam_c) {
	//						int c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
	//						for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc) {
	//							int c = matchmap[b + b_start][cam_c][match_bc];
	//							if (c < 0) {
	//								break;
	//							}
	//							lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];								
	//							if (std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c))
	//								for (int cam_d = cam_c + 1; cam_d < frame_group.size(); ++cam_d) {
	//									int d_start = (cam_d != 0 ? num_particles[cam_d - 1] : 0);
	//									for (int match_cd = 0; match_cd < NUM_MATCHES; ++match_cd) {
	//										int d = matchmap[c + c_start][cam_d][match_cd];
	//										if (d < 0) {
	//											break;
	//										}
	//										lpt::ParticleImage::Ptr Pd = frame_group[cam_d].particles[d];											
	//										if (std::count(matchmap[a + a_start][cam_d].begin(), matchmap[a + a_start][cam_d].end(), d) && std::count(matchmap[b + b_start][cam_d].begin(), matchmap[b + b_start][cam_d].end(), d)) {
	//											lpt::Match::Ptr newmatch = lpt::Match::create();
	//											newmatch->addParticle(Pa, cam_a);
	//											newmatch->addParticle(Pb, cam_b);
	//											newmatch->addParticle(Pc, cam_c);
	//											newmatch->addParticle(Pd, cam_d);
	//											MatchAndRpjErrs mae; //mahdi
	//											mae.match = newmatch;
	//											Correspondence::reprojection_error_calc(newmatch, mae.p3drec, mae.reprojectionErr);
	//											matchAndErr.push_back(mae);												
	//										}
	//									}
	//								}
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//	//minErrMatches = Correspondence::findMinErrMatch(matchAndErr);
	//	vector <lpt::Match::Ptr> minErrMatchesReturn = findMinErrMatchReturn(matchAndErr, params.threshold_distance);
	//	std::move(minErrMatchesReturn.begin(), minErrMatchesReturn.end(), std::back_inserter(matches));
	//}
	////----------------


void PointMatcherCUDA::find3WayMatches_mahdi(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
	vector<int> num_particles(frame_group.size());
	num_particles[0] = frame_group[0].particles.size();
	for (int i = 1; i < frame_group.size(); ++i) {
		num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];
	}

	for (int cam_a = 0; cam_a < frame_group.size() - 2; ++cam_a) { // Ensure there are at least two more cameras
		int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
		for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
			lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
			if (!Pa->is_4way_matched) {
				for (int cam_b = cam_a + 1; cam_b < frame_group.size() - 1; ++cam_b) { // Ensure there is at least one more camera
					int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
					for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { // Loop through all A,B matches
						int b = matchmap[a + a_start][cam_b][match_ab];
						if (b < 0)
							break;
						lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];
						if (!Pb->is_4way_matched) {
							for (int cam_c = cam_b + 1; cam_c < frame_group.size(); ++cam_c) { // Check for a valid third camera
								int c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
								for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc) {
									int c = matchmap[b + b_start][cam_c][match_bc];
									if (c < 0)
										break;

									lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];
									if (!Pc->is_4way_matched && std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c)) {
										// Verify all three particles are unmatched then create a new match
										if (!Pa->is_4way_matched && !Pb->is_4way_matched && !Pc->is_4way_matched) {
											lpt::Match::Ptr newmatch = lpt::Match::create();
											newmatch->addParticle(Pa, cam_a);
											newmatch->addParticle(Pb, cam_b);
											newmatch->addParticle(Pc, cam_c);
											matches.push_back(newmatch);

											// Mark as matched
											Pa->is_4way_matched = true;
											Pb->is_4way_matched = true;
											Pc->is_4way_matched = true;

											// Break out of the loops since match is found
											match_ab = NUM_MATCHES;
											match_bc = NUM_MATCHES;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	//vector<int> num_particles(frame_group.size());
	//num_particles[0] = frame_group[0].particles.size();
	//for (int i = 1; i < frame_group.size(); ++i)
	//	num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];
	//for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
	//	int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
	//	for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
		//		lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
		//		if (!Pa->is_4way_matched)
		//			for (int cam_b = cam_a + 1; cam_b < frame_group.size() - 1; ++cam_b) {
		//				int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
		//				for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
		//					int b = matchmap[a + a_start][cam_b][match_ab];
		//					if (b < 0)
		//						break;
		//					lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];
		//					if (!Pb->is_4way_matched)
		//						for (int cam_c = cam_b + 1; cam_c < frame_group.size()-1; ++cam_c) {
		//							vector<lpt::Match::Ptr> matches4way;
		//							int c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
		//							for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc) {
		//								int c = matchmap[b + b_start][cam_c][match_bc];
		//								if (c < 0)
		//									break;
		//								lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];
		//								if (!Pc->is_4way_matched && std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c))
		//									if (!Pa->is_4way_matched && !Pb->is_4way_matched && !Pc->is_4way_matched) {												
		//										lpt::Match::Ptr newmatch = lpt::Match::create();
		//										newmatch->addParticle(Pa, cam_a);
		//										newmatch->addParticle(Pb, cam_b);
		//										newmatch->addParticle(Pc, cam_c);
		//										matches4way.push_back(std::move(newmatch));
		//										Pa->is_4way_matched = true;
		//										Pb->is_4way_matched = true;
		//										Pc->is_4way_matched = true;
		//										match_ab = NUM_MATCHES;
		//										match_bc = NUM_MATCHES;
		//									}
		//							}
		//							std::move(matches4way.begin(), matches4way.end(), std::back_inserter(matches));
		//						}
		//				}
		//			}
		//	}
		//}
	}
		
		/*vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];
		int num_cameras = frame_group.size();
		for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched) {
					for (int cam_b = cam_a + 1; cam_b < frame_group.size() - 1; ++cam_b) {
						int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
						for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
							int b = matchmap[a + a_start][cam_b][match_ab];
							if (b < 0)
								break;
							lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];
							if (!Pb->is_4way_matched) {
								for (int cam_c = cam_b + 1; cam_c < frame_group.size(); ++cam_c) {
									int c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
									for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc) {
										int c = matchmap[b + b_start][cam_c][match_bc];
										if (c < 0)
											break;
										lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];
										if (!Pc->is_4way_matched && std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c)) {
											lpt::Match::Ptr newmatch = lpt::Match::create();
											newmatch->addParticle(Pa, cam_a);
											newmatch->addParticle(Pb, cam_b);
											newmatch->addParticle(Pc, cam_c);
											matches.push_back(std::move(newmatch));
											Pa->is_4way_matched = true;
											Pb->is_4way_matched = true;
											Pc->is_4way_matched = true;
											match_ab = NUM_MATCHES;
											match_bc = NUM_MATCHES;
											cam_b = num_cameras;
											cam_c = num_cameras;
										}
									}
								}
							}
						}
					}
				}
			}
		}*/

	//mahdi modify method find2WayMathces at dec 10: it does not compile, but I checked, no diff in code between this and the one clled wo clear
	void PointMatcherCUDA::find2WayMathces(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
		cout << " mahdi this is under find2way in cuda without clear and no rpj err check, temp data commented " << endl;
		vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		int num_cameras = frame_group.size();

		for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched) {
					for (int cam_b = cam_a + 1; cam_b < frame_group.size(); ++cam_b) {
						int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
						for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
							int b = matchmap[a + a_start][cam_b][match_ab];
							if (b < 0)
								break;
							lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];

							lpt::Match::Ptr newmatch = lpt::Match::create();

							vector<vector<float>> temp_data;
							vector<lpt::Match::Ptr> temp_pairs_vec;
							vector<float> temp_transformed_data(10, 0);

							newmatch->addParticle(Pa, cam_a);
							newmatch->addParticle(Pb, cam_b);
							matches.push_back(std::move(newmatch)); // added by mahdi

							//// lines below commented by mahdi till the end

							//temp_pairs_vec.push_back(newmatch);
							//construct_pair_feature(temp_pairs_vec, frame_group, temp_data);

							////for (int i = 0; i < 30; i++) {
							////	temp_data[0][i] = (temp_data[0][i] - x_train_mean[i]) / x_train_std[i];
							////}

							////// temp_data * pca_paras
							////for (int i = 0; i < 10; i++) {
							////	for (int j = 0; j < 30; j++) {
							////		temp_transformed_data[i] += temp_data[0][j] * pca_parameters[j][i];
							////	}
							////}

							////float prob = 0.0f; float temp = 0.0f;
							////// perform LR prob
							////for (int i = 0; i < 10; i++) {
							////	temp += temp_transformed_data[i] * lr_data[i];
							////}

							////temp += lr_data[10];

							////prob = 1 / (1 + exp(-temp));

							//float prob[1];
							//prob[0] = 0.0f;
							//float* prob_device = NULL;
							//gpuErrchk(cudaMalloc((void**)(&prob_device), 1 * sizeof(float)));
							//gpuErrchk(cudaMemcpy(prob_device, prob, sizeof(float) * 1, cudaMemcpyHostToDevice));

							//float* temp_device = NULL;
							//gpuErrchk(cudaMalloc((void**)(&temp_device), 30 * sizeof(float)));
							//gpuErrchk(cudaMemcpy(temp_device, temp_data[0].data(), sizeof(float) * 30, cudaMemcpyHostToDevice));

							//dim3 dimBlock(32, 1, 1);
							//dim3 dimGrid(1, 1, 1);
							////test_kernel_corresponde << <dimGrid, dimBlock, 0 >> > (temp_device, prob_device);

							//gpuErrchk(cudaDeviceSynchronize());
							//gpuErrchk(cudaPeekAtLastError());

							//gpuErrchk(cudaMemcpy(prob, prob_device, 1 * sizeof(float), cudaMemcpyDeviceToHost));

							////cout << "prob is: " << prob[0] << endl;

							//if (prob[0] > 0.5) {
							//	matches.push_back(std::move(newmatch));

							//	Pa->is_4way_matched = true;
							//	Pb->is_4way_matched = true;
							//}
							//match_ab = NUM_MATCHES;
							//cam_b = num_cameras;

							//// lines above commented by mahdi 

						}
					}
				}
			}
		}

	}


	void PointMatcherCUDA::find2WayMathces_wo_clear(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
		//cout << " mahdi this is under find2way in cuda without clear and no rpj err check " << endl;
		vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		//cout << " mahdi under find2way matches_wo_clear in cuda" << endl;
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		int num_cameras = frame_group.size();

		vector<vector<float>> temp_data;
		//vector<torch::jit::IValue> inputs;
		vector<lpt::Match::Ptr> temp_pairs_vec;

		for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched) {
					for (int cam_b = cam_a + 1; cam_b < frame_group.size(); ++cam_b) {
						int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
						for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
							int b = matchmap[a + a_start][cam_b][match_ab];
							if (b < 0)
								break;
							lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];

							lpt::Match::Ptr newmatch = lpt::Match::create();
							newmatch->addParticle(Pa, cam_a);
							newmatch->addParticle(Pb, cam_b);
							matches.push_back(std::move(newmatch)); // added by mahdi
							//temp_pairs_vec.push_back(newmatch);

						}
					}
				}
			}
		}

		//construct_pair_feature(temp_pairs_vec, frame_group, temp_data);

		//int rows = temp_data.size();
	}
	/*if (rows != 0) {
		int cols = temp_data[0].size();
		auto options = torch::TensorOptions().dtype(at::kFloat);

		if (cols) {
			at::Tensor temp_data_tensor = torch::ones({ rows, cols }, options);

			for (int i = 0; i < rows; i++)
				temp_data_tensor.slice(0, i, i + 1) = torch::from_blob(temp_data[i].data(), { cols }, options);

			temp_data_tensor.to(at::kCUDA);
			temp_data_tensor = (temp_data_tensor - mean_tensor) / std_tensor;
			temp_data_tensor = at::matmul(temp_data_tensor, pca_tensor).to(at::kCUDA);

			vector<torch::jit::IValue> inputs;
			inputs.push_back(temp_data_tensor);

			torch_module.eval();
			torch::NoGradGuard no_grad;

			at::Tensor output = torch_module.forward(inputs).toTensor();
			output = torch::sigmoid(output);
			output = torch::round(output);

			output = output.to(at::kCPU);

			auto output_a = output.accessor<float, 2>();

			int temp_count = 0;
			for (int i = 0; i < output_a.size(0); i++) {
				if ((int)output_a[i][0] == 1) {
					if (!temp_pairs_vec[i]->particles[0].first->is_4way_matched && !temp_pairs_vec[i]->particles[1].first->is_4way_matched) {
						matches.push_back(temp_pairs_vec[i]);
						temp_pairs_vec[i]->particles[0].first->is_4way_matched = true;
						temp_pairs_vec[i]->particles[1].first->is_4way_matched = true;
					}
					temp_count++;
				}
			}
		}
	}
}*/

	void PointMatcherCUDA::find3WayMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
		//cout << " mahdi this is under find3way in cuda with clear and no rpj err check " << endl;
		vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		int num_cameras = frame_group.size();
		matches.clear();

		for (int cam_a = 0; cam_a < frame_group.size() - 2; ++cam_a) {
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched) {
					for (int cam_b = cam_a + 1; cam_b < frame_group.size() - 1; ++cam_b) {
						int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
						for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
							int b = matchmap[a + a_start][cam_b][match_ab];
							if (b < 0)
								break;
							lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];

							if (!Pb->is_4way_matched) {
								for (int cam_c = cam_b + 1; cam_c < frame_group.size(); ++cam_c) {
									int c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
									for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc) {
										int c = matchmap[b + b_start][cam_c][match_bc];
										if (c < 0)
											break;

										lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];

										if (!Pc->is_4way_matched && std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c)) {
											lpt::Match::Ptr newmatch = lpt::Match::create();
											newmatch->addParticle(Pa, cam_a);
											newmatch->addParticle(Pb, cam_b);
											newmatch->addParticle(Pc, cam_c);

											matches.push_back(std::move(newmatch));

											Pa->is_4way_matched = true;
											Pb->is_4way_matched = true;
											Pc->is_4way_matched = true;

											match_ab = NUM_MATCHES;
											match_bc = NUM_MATCHES;

											cam_b = num_cameras;
											cam_c = num_cameras;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		//cout << matches.size() << endl;
	}

	void PointMatcherCUDA::findEpipolarMatchesStreams(lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap) {
		thrust::fill(matches2way_d.begin(), matches2way_d.end(), match_initializer);
		num_particles_h[0] = frame_group[0].particles.size();
		for (int p = 0; p < frame_group[0].particles.size(); ++p) {
			particles_x_h[p] = static_cast<float>(frame_group[0].particles[p]->x);
			particles_y_h[p] = static_cast<float>(frame_group[0].particles[p]->y);
		}

		int max_particles = num_particles_h[0];
		for (int i = 1; i < frame_group.size(); ++i) {
			num_particles_h[i] = frame_group[i].particles.size() + num_particles_h[i - 1];
			for (int p = 0; p < frame_group[i].particles.size(); ++p) {
				particles_x_h[num_particles_h[i - 1] + p] = static_cast<float>(frame_group[i].particles[p]->x);
				particles_y_h[num_particles_h[i - 1] + p] = static_cast<float>(frame_group[i].particles[p]->y);
			}
			if (frame_group[i].particles.size() > max_particles)
				max_particles = frame_group[i].particles.size();
		}

		cudaMemcpyAsync(thrust::raw_pointer_cast(&num_particles_d[0]), thrust::raw_pointer_cast(&num_particles_h[0]), num_particles_h.size() * sizeof(int), cudaMemcpyHostToDevice, streams[0]);
		cudaMemcpyAsync(thrust::raw_pointer_cast(&particles_x_d[0]), thrust::raw_pointer_cast(&particles_x_h[0]), *num_particles_h.rbegin() * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
		cudaMemcpyAsync(thrust::raw_pointer_cast(&particles_y_d[0]), thrust::raw_pointer_cast(&particles_y_h[0]), *num_particles_h.rbegin() * sizeof(float), cudaMemcpyHostToDevice, streams[0]);

		int num_pairs = camera_pairs_h.size();
		dim3 dimblock(128, 1, 1);
		dim3 dimgrid(static_cast<unsigned int>(num_pairs / streams.size()), (static_cast<unsigned int>(max_particles) / dimblock.x) + 1, 1);

		num_matches_h[0] = frame_group[camera_pairs_h[0].cam_b_id].particles.size();
		for (int i = 1; i < this->camera_pairs_h.size(); ++i) {
			num_matches_h[i] = frame_group[camera_pairs_h[i].cam_b_id].particles.size() + num_matches_h[i - 1];
		}

		cudaMemcpyAsync(thrust::raw_pointer_cast(&num_matches_d[0]), thrust::raw_pointer_cast(&num_matches_h[0]), num_matches_h.size() * sizeof(int), cudaMemcpyHostToDevice, streams[0]);

		for (int i = 0; i < streams.size(); i++)
			calcEpipolarResidualAllInOneStreams_kernel << < dimgrid, dimblock, 0, streams[i] >> > (i * dimgrid.x, params.match_threshold, particles_x_d, particles_y_d, num_particles_d, matches2way_d, num_matches_d);

		for (int i = 0; i < streams.size(); i++) {
			int index = (i == 0 ? 0 : num_matches_h[i * dimgrid.x - 1]);
			int nbytes = (i == 0 ? num_matches_h[dimgrid.x - 1] : num_matches_h[(i + 1) * dimgrid.x - 1] - num_matches_h[i * dimgrid.x - 1]) * sizeof(MatchIDs);
			cudaMemcpyAsync(thrust::raw_pointer_cast(&matches2way_h[index]), thrust::raw_pointer_cast(&matches2way_d[index]), nbytes, cudaMemcpyDeviceToHost, streams[i]);
		}

		int match_overload = 0;
		for (unsigned int i = 0; i < streams.size(); ++i) {
			cudaStreamSynchronize(streams[i]);
			for (unsigned int p = i * dimgrid.x; p < (i + 1) * dimgrid.x; ++p) {
				int match_id = (p == 0 ? 0 : num_matches_h[p - 1]);
				int cam_b = camera_pairs_h[p].cam_b_id;
				int cam_a = camera_pairs_h[p].cam_a_id;
				int b_end = num_particles_h[cam_b];
				int b_start = (cam_b != 0 ? num_particles_h[cam_b - 1] : 0);
				int a_start = (cam_a != 0 ? num_particles_h[cam_a - 1] : 0);
				for (int b_id = b_start; b_id < b_end; ++b_id, ++match_id) {
					for (int m = 0; m < NUM_MATCHES; ++m) {
						int a_id = matches2way_h[match_id].ids[m];
						if (a_id >= 0) {
							//matchmap[b_id][cam_a][m] = a_id - a_start;
							//matchmap[a_id][cam_b][m] = b_id - b_start; 
							auto itb = std::find(matchmap[b_id][cam_a].begin(), matchmap[b_id][cam_a].end(), -1);
							auto ita = std::find(matchmap[a_id][cam_b].begin(), matchmap[a_id][cam_b].end(), -1);
							if (itb != matchmap[b_id][cam_a].end() && ita != matchmap[a_id][cam_b].end()) {
								*itb = a_id - static_cast<int>(a_start);
								*ita = static_cast<int>(b_id - b_start);
							}
							else
								match_overload++;
						}
						else
							break;
					}
				}
			}
		}

		if (match_overload > 0)
			;//cout << "WARNING: MORE MATCHES THAN ARRAY SIZE NUM_MATCHES: total overload = " << match_overload << endl;
	}

	void PointMatcherCUDA::findEpipolarMatchesManyThreads(lpt::ImageFrameGroup& frame_group) {

		num_particles_h[0] = frame_group[0].particles.size();
		for (int p = 0; p < frame_group[0].particles.size(); ++p) {
			particles_x_h[p] = static_cast<float>(frame_group[0].particles[p]->x);
			particles_y_h[p] = static_cast<float>(frame_group[0].particles[p]->y);
		}

		int max_particles = num_particles_h[0];
		for (int i = 1; i < frame_group.size(); ++i) {
			num_particles_h[i] = frame_group[i].particles.size() + num_particles_h[i - 1];
			for (int p = 0; p < frame_group[i].particles.size(); ++p) {
				particles_x_h[num_particles_h[i - 1] + p] = static_cast<float>(frame_group[i].particles[p]->x);
				particles_y_h[num_particles_h[i - 1] + p] = static_cast<float>(frame_group[i].particles[p]->y);
			}
			if (frame_group[i].particles.size() > max_particles)
				max_particles = frame_group[i].particles.size();
		}

		cudaMemcpyAsync(thrust::raw_pointer_cast(&num_particles_d[0]), thrust::raw_pointer_cast(&num_particles_h[0]), num_particles_h.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(thrust::raw_pointer_cast(&particles_x_d[0]), thrust::raw_pointer_cast(&particles_x_h[0]), *num_particles_h.rbegin() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(thrust::raw_pointer_cast(&particles_y_d[0]), thrust::raw_pointer_cast(&particles_y_h[0]), *num_particles_h.rbegin() * sizeof(float), cudaMemcpyHostToDevice);

		int num_pairs = camera_pairs_h.size();
		dim3 dimblock(256);
		dim3 dimgrid(static_cast<unsigned int>(num_pairs), (static_cast<unsigned int>(max_particles) / dimblock.x) + 1);

		thrust::host_vector<int> num_lines_h(camera_pairs_h.size(), 0);
		num_lines_h[0] = frame_group[camera_pairs_h[0].cam_b_id].particles.size();
		for (int i = 1; i < this->camera_pairs_h.size(); ++i)
			num_lines_h[i] = frame_group[camera_pairs_h[i].cam_b_id].particles.size() + num_lines_h[i - 1];

		thrust::device_vector<int> num_lines_d = num_lines_h;
		thrust::device_vector<float> lines_x(*num_lines_h.rbegin(), 0.f);
		thrust::device_vector<float> lines_y(*num_lines_h.rbegin(), 0.f);
		thrust::device_vector<float> lines_z(*num_lines_h.rbegin(), 0.f);

		//	lines_x.resize( *num_lines_h.rbegin(), 0.f );
		//	lines_y.resize( *num_lines_h.rbegin(), 0.f );
		//	lines_z.resize( *num_lines_h.rbegin(), 0.f );	

		calcEpipolarLines_kernel << < dimgrid, dimblock >> > (particles_x_d, particles_y_d, num_particles_d, lines_x, lines_y, lines_z, num_lines_d);

		num_matches_h[0] = frame_group[camera_pairs_h[0].cam_b_id].particles.size();
		for (int i = 1; i < this->camera_pairs_h.size(); ++i) {
			num_matches_h[i] = frame_group[camera_pairs_h[i].cam_b_id].particles.size() + num_matches_h[i - 1];
		}

		cudaMemcpyAsync(thrust::raw_pointer_cast(&num_matches_d[0]), thrust::raw_pointer_cast(&num_matches_h[0]), num_matches_h.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaStreamSynchronize(0);

		dim3 dimblock2(512, 1, 1);
		dim3 dimgrid2(static_cast<unsigned int>(*num_lines_h.rbegin()), (static_cast<unsigned int>(max_particles) / dimblock2.x) + 1, 1);
		//cout <<"K2 Grid = " << dimgrid2.x << " x " << dimgrid2.y << " x " << dimgrid2.z << endl;
		//cout <<"K2 Block = " << dimblock2.x << " x " << dimblock2.y << " x " << dimblock2.z << endl;

		calcEpipolarResiduals_kernel << < dimgrid2, dimblock2 >> > (params.match_threshold, particles_x_d, particles_y_d, num_particles_d, lines_x, lines_y, lines_z, num_lines_d, matches2way_d, num_matches_d);

		thrust::copy(matches2way_d.begin(), matches2way_d.begin() + *num_matches_h.rbegin(), matches2way_h.begin());

	}

}
