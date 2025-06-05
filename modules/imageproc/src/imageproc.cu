/**
 * @file imageproc.cpp
 * Image processing module definition
 */

#include "imageproc.hpp"
#include "stdio.h"
//#include "C:/libsvm/SVM_example-master/svm.cpp"

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
//{
//	if (code != cudaSuccess)
//	{
//		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//		if (abort) exit(code);
//	}
//}
//
//struct is_true : thrust::unary_function<bool, bool>
//{
//	__host__ __device__ bool operator()(const bool &x) {
//		return x;
//	}
//};
//
//__global__ void test_kernel(bool* isGood, int size) {
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//	//if (i < size) {
//	//	if (i < (int)(size / 2))
//	//		isGood[i] = true;
//	//	else
//	//		isGood[i] = false;
//	//}
//	isGood[i] = true;
//
//}
//
//__global__ void calculateDistances_centroids(float2* corners_1, float2* corners_2, float2* centroids, int size_1, int size_2, int size_3, float* distances_1, float* distances_2, float threshold, bool* isGood) {
//	int j = blockDim.x * blockIdx.x + threadIdx.x;
//	int i = blockDim.y * blockIdx.y + threadIdx.y;
//
//	if (i < size_1 && j < size_2) {
//		int index = j + i * (size_2);
//
//		float dist = (corners_1[i].x - corners_2[j].x) * (corners_1[i].x - corners_2[j].x) + (corners_1[i].y - corners_2[j].y) * (corners_1[i].y - corners_2[j].y);
//		dist = sqrt(dist);
//
//		distances_1[index] = dist;
//	}
//
//	// calculate the distances_2
//	if (i < size_1 && j < size_3) {
//		int index = j + i * (size_3);
//
//		float dist = (corners_1[i].x - centroids[j].x) * (corners_1[i].x - centroids[j].x) + (corners_1[i].y - centroids[j].y) * (corners_1[i].y - centroids[j].y);
//		dist = sqrt(dist);
//
//		distances_2[index] = dist;
//	}
//
//	__syncthreads();
//
//	if (j == 0 && i < size_1) {
//		float* start_1 = &distances_1[i * size_2];
//		float* end_1 = &distances_1[(i + 1) * size_2];
//		float* smallestThrust_1 = thrust::min_element(thrust::device, start_1, end_1);
//
//		float* start_2 = &distances_2[i * size_3];
//		float* end_2 = &distances_2[(i + 1) * size_3];
//		float* smallestThrust_2 = thrust::min_element(thrust::device, start_2, end_2);
//
//		if (*smallestThrust_1 < threshold || *smallestThrust_2 < threshold / 2)
//			isGood[i] = true;
//		else
//			isGood[i] = false;
//
//		//if (*smallestThrust_1 < threshold)
//		//	isGood[i] = true;
//		//else
//		//	isGood[i] = false;
//
//		start_1 = NULL;
//		end_1 = NULL;
//		smallestThrust_1 = NULL;
//
//		start_2 = NULL;
//		end_2 = NULL;
//		smallestThrust_2 = NULL;
//	}
//}
//
//__global__ void calculateDistances(float2* corners_1, float2* corners_2, int size_1, int size_2, float* distances, float threshold, bool* isGood) {
//	int j = blockDim.x * blockIdx.x + threadIdx.x;
//	int i = blockDim.y * blockIdx.y + threadIdx.y;
//
//	if (i < size_1 && j < size_2) {
//		int index = j + i * (size_2);
//
//		float dist = (corners_1[i].x - corners_2[j].x) * (corners_1[i].x - corners_2[j].x) + (corners_1[i].y - corners_2[j].y) * (corners_1[i].y - corners_2[j].y);
//		dist = sqrt(dist);
//
//		distances[index] = dist;
//	}
//
//	__syncthreads();
//
//	if (j == 0 && i < size_1) {
//		float* start = &distances[i * size_2];
//		float* end = &distances[(i + 1) * size_2];
//
//		float* smallestThrust = thrust::min_element(thrust::device, start, end);
//
//		if (*smallestThrust < threshold)
//			isGood[i] = true;
//		else
//			isGood[i] = false;
//
//		start = NULL;
//		end = NULL;
//		smallestThrust = NULL;
//	}
//}
//
////thrust::host_vector<float2> filter_closest_corners(thrust::device_vector<float2>& currentCorners, thrust::device_vector<float2>& previousCorners, thrust::host_vector<float2>& retCorners, float threshold, cudaStream_t& s) {
//void filter_closest_corners(thrust::device_vector<float2>& currentCorners, thrust::device_vector<float2>& previousCorners, thrust::host_vector<float2>& retCorners, float threshold, cudaStream_t& s) {
//	int size_1 = currentCorners.size();
//	int size_2 = previousCorners.size();
//
//	thrust::device_vector<bool> isGoodGpu(size_1);
//
//	// test to fill isGoodGpu true
//	thrust::fill(isGoodGpu.begin(), isGoodGpu.end(), true);
//
//	//bool* isGoodGpuPtr = thrust::raw_pointer_cast(isGoodGpu.data());
//
//	//float* distances = NULL;
//	//gpuErrchk(cudaMalloc((void**)(&distances), size_1 * size_2 * sizeof(float)));
//
//	//dim3 dimBlock(32, 32, 1);
//	//dim3 dimGrid((int)(size_1 / 32 + 1), (int)(size_2 / 32 + 1), 1);
//
//	//float2* currentCornersPtr = thrust::raw_pointer_cast(currentCorners.data());
//	//float2* previousCornersPtr = thrust::raw_pointer_cast(previousCorners.data());
//
//	//calculateDistances << <dimGrid, dimBlock, 0, s >> > (currentCornersPtr, previousCornersPtr, size_1, size_2, distances, threshold, isGoodGpuPtr);
//	//gpuErrchk(cudaPeekAtLastError());
//	//
//	//gpuErrchk(cudaStreamSynchronize(s));
//
//	thrust::device_vector<float2> tempCorners(size_1);
//	thrust::device_vector<float2> trueCorners(size_1);
//
//	// copy from gpumat to thrust vector
//	thrust::copy(thrust::cuda::par.on(s), currentCorners.begin(), currentCorners.begin(), tempCorners.begin());
//
//	thrust::device_vector<float2>::iterator iter_end = thrust::copy_if(thrust::cuda::par.on(s), tempCorners.begin(), tempCorners.end(), isGoodGpu.begin(), trueCorners.begin(), is_true());
//	
//	gpuErrchk(cudaStreamSynchronize(s));
//
//	int len = iter_end - trueCorners.begin();
//
//	std::cout << " len is: " << len << std::endl;
//
//	retCorners.resize(len);
//	thrust::copy(trueCorners.begin(), iter_end, retCorners.begin());
//
//	//gpuErrchk(cudaStreamSynchronize(s));
//	//gpuErrchk(cudaFree(distances));
//	//currentCornersPtr = NULL;
//	//previousCornersPtr = NULL;
//}

namespace lpt {

using namespace std;

void undistortPoints(const lpt::Camera& camera, lpt::ImageFrame& frame) {
	if (!frame.particles.empty()) {
		vector<cv::Point2d> image_points(frame.particles.size());

		for (int j = 0; j < frame.particles.size(); ++j) {
			image_points[j].x = frame.particles[j]->x;
			image_points[j].y = frame.particles[j]->y;
		}		
		cv::undistortPoints(image_points, image_points, camera.getCameraMatrix(), camera.getDistCoeffs(), cv::Mat(), camera.getCameraMatrix());		
		for (int j = 0; j < frame.particles.size(); ++j) { 
			frame.particles[j]->x = image_points[j].x;
			frame.particles[j]->y = image_points[j].y;	
		}
	}
}

void filterParticles(lpt::ImageFrame& frame,
	double minRadius, double maxRadius,
	double minX, double maxX, double minY, double maxY) {
	frame.image = cv::Mat::zeros(cv::Size(frame.image.cols, frame.image.rows), CV_8UC1);
	// Prepare a container to hold the filtered particles
	std::vector<lpt::ParticleImage::Ptr> filtered_particles;
	// Iterate over all particles in the frame
	for (const auto& particle : frame.particles) {
		// Check if the particle meets the radius and coordinate conditions
		if (particle->radius >= minRadius && particle->radius <= maxRadius &&
			particle->x >= minX && particle->x <= maxX &&
			particle->y >= minY && particle->y <= maxY) {
			// Add the particle to the filtered list
			filtered_particles.push_back(particle);
			cv::Point center(round(particle->x), round(particle->y));
			int radius = static_cast<int>(particle->radius);
			int intensity = 255;  // Max intensity for CV_8UC1
			//intensity = particle->intensity;
			cv::circle(frame.image, center, radius, cv::Scalar(intensity), -1);
		}
		// below is using something like VFcreateimage but above also works
//			if (particle->radius >= 1) {
//				//cv::Point center(static_cast<int>(x), static_cast<int>(y));
////cv::circle(frame.image, center, static_cast<int>(r), cv::Scalar(I, I, I), -1, 8);
//				cv::Point center(std::ceil(particle->x), std::ceil(particle->y));  // Always round down/up
//				cv::circle(frame.image, center, std::floor(particle->radius), 
//					cv::Scalar(particle->intensity, particle->intensity, particle->intensity), -1, 8);
//
//			}
//			else {
//				if (static_cast<int>(particle->y) >= 0 && static_cast<int>(particle->y) < frame.image.rows && static_cast<int>(particle->x) >= 0 && static_cast<int>(particle->x) < frame.image.cols)
//					frame.image.at<uchar>(static_cast<int>(particle->y), static_cast<int>(particle->x)) = static_cast<unsigned char>(particle->intensity);
//			}
	}
	// Replace the original particle list with the filtered list
	frame.particles = filtered_particles;
}

/*
void undistortAndFilterParticles(const lpt::Camera& camera, lpt::ImageFrame& frame,
	double minRadius, double maxRadius,
	double minX, double maxX, double minY, double maxY) {
	if (!frame.particles.empty()) {
		std::vector<cv::Point2f> image_points;
		std::vector<lpt::ParticleImage::Ptr> filtered_particles;

		// Collect valid points for undistortion
		for (auto& particle : frame.particles) {
			if (particle->radius >= minRadius && particle->radius <= maxRadius &&
				particle->x >= minX && particle->x <= maxX &&
				particle->y >= minY && particle->y <= maxY) {
				image_points.emplace_back(cv::Point2f(static_cast<float>(particle->x), static_cast<float>(particle->y)));
				filtered_particles.push_back(particle);
			}
		}

		// Apply undistortion if there are any valid points left
		if (!image_points.empty()) {
			cv::undistortPoints(image_points, image_points, camera.getCameraMatrix(), camera.getDistCoeffs(), cv::Mat(), camera.getCameraMatrix());

			// Update the coordinates in the filtered particles list
			for (size_t j = 0; j < filtered_particles.size(); ++j) {
				filtered_particles[j]->x = image_points[j].x;
				filtered_particles[j]->y = image_points[j].y;
			}
		}

		// Replace the original particle list with the filtered and updated list
		frame.particles = filtered_particles;
	}
}
*/

ImageProcess::~ImageProcess()
{
    cout << "ImageProcess destructed" << endl;
}

ImageProcessor::ImageProcessor()
{
    cout << "ImageProcessor constructed" << endl;
    cout << "OpenCV version : " << CV_VERSION << endl;
}

void ImageProcessor::processImage(cv::Mat &image)
{
    for(int i = 0; i < m_processes.size(); ++i) {
        m_processes[i]->process(image);
    }
}

void ImageProcessor::addControls()
{
    cout << "\t--Adding Image Processor Controls to window: " << endl;
    for(int i = 0; i < m_processes.size(); ++i) {
        if (m_processes[i])
            m_processes[i]->addControls();
        else
            cout << "process " << i << " invalid pointer" << endl;
    }
}

void ImageProcessor::addProcess(ImageProcess::Ptr process)
{
    m_processes.push_back(process);
    cout << "Image process added " << endl;
}

SubtractBackground::SubtractBackground()
{
    cout << "SubtractBackground created" << endl;
}

void SubtractBackground::addControls()
{
    cout << "Add controls for background subtraction" << endl;
}

SubtractBackground::~SubtractBackground()
{
    cout << "SubtractBackgournd destructed" << endl;
}

Resize::Resize(cv::Size s, double x_scale, double y_scale, int type)
	:size(s), xScale(x_scale), yScale(y_scale), srcType(type) {
	cout << "Resize contructed" << endl;
}

void Resize::addControls() {
	cout << "\t\tAdded Resize process to Control Window" << endl;
}

Resize::~Resize() {
	cout << "Resize destructed" << endl;
}

Threshold::Threshold(int threshold, int max_threshold)
  : m_threshold(threshold), m_max_threshold(max_threshold)
{
    cout << "Threshold constructed" << endl;
}

void Threshold::addControls()
{
    cv::createTrackbar("ThresholdImage", string(), &m_threshold, m_max_threshold, 0, 0 );
    cout << "\t\tAdded Threshold process to Control Window" << endl;
}

Threshold::~Threshold()
{
    cout << "Threshold destructed" << endl;
}

Erode::Erode(int iterations, int max_iterations)
  : m_iterations(iterations), m_max_iterations(max_iterations)
{
    cout << "Erode constructed" << endl;
}

void Erode::addControls()
{
    cv::createTrackbar("Erode iterations", string(), &m_iterations, m_max_iterations, 0, 0 );
}

Erode::~Erode()
{
    cout << "Erode desturcted" << endl;
}

EqualizeHistogram::EqualizeHistogram()
{
    cout << "EqualizeHistogram constructed" << endl;
}

void EqualizeHistogram::addControls()
{
    cout << "Add controls for EqualizeHistogram" << endl;
}

EqualizeHistogram::~EqualizeHistogram()
{
    cout << "EqualizeHistogram destructed" << endl;
}

Dilate::Dilate(int iterations, int max_iterations)
  : m_iterations(iterations), m_max_iterations(max_iterations)
{
    cout << "Dilate constructed" << endl;
}

void Dilate::addControls()
{
    cv::createTrackbar("Dilate iterations", string(), &m_iterations, m_max_iterations, 0, 0 );
}

Dilate::~Dilate()
{
    cout << "Dilate destructed" << endl;
}

GaussianBlur::GaussianBlur(int kernel_size, double sigma1, double sigma2, int boarder)
  : m_kernel_size(kernel_size), m_sigma1(sigma1), m_sigma2(sigma2), m_boarder_type(boarder)
{
    cout << "Gaussian blur constructed" << endl;
}

void GaussianBlur::addControls()
{
    //cv::createTrackbar("GaussBlur ksize", window_name, &kernel_size, 9, 0, 0 );
    cout << "\t\tAdded GaussianBlur to Control Window" << endl;
}

GaussianBlur::~GaussianBlur()
{
    cout << "Gaussian blur destructed" << endl;
}

void Detector::drawResult(ImageFrame &frame)
{
    //cv::cvtColor(frame.image, frame.image, CV_GRAY2BGR);
    for (int i = 0; i < frame.particles.size(); i++) {
		auto particle = frame.particles[i];
		//cv::circle(frame.image, cv::Point(particle->x, particle->y), static_cast<int>(particle->radius), 255, 1);
        cv::circle( frame.image, cv::Point(particle->x, particle->y), 
			particle->radius, 255, 1); //mahdi circle color was 200 and radius thickness was 1
		////mahdi draw rectangle instead of circle, inputs: topleft, bottom right, color, thickness, linetype, shift
		//int rectangeOffset = 3;
		//cv::Point topleft(static_cast<int>(particle->x - rectangeOffset), static_cast<int>(particle->y - rectangeOffset));
		//cv::Point bottomRight(static_cast<int>(particle->x + rectangeOffset), static_cast<int>(particle->y + rectangeOffset));
		//cv::rectangle(frame.image, topleft, bottomRight, 255, 1, cv::LINE_8, 0);
    }
}

Detector::~Detector()
{
    cout << "Detector destructed" << endl;
}

FindContoursDetector::FindContoursDetector()
{
	is_gfft = false;
    cout << "Find Contours detector constructed" << endl;
}


void FindContoursDetector::addControls()
{
    cout << "\t--Adding Find Contours Detector Controls to window: " << endl;
    cv::createTrackbar("Min Area", string(), &params.min_contour_area, 500, 0, 0 );
    cv::createTrackbar("Max Area", string(), &params.max_contour_area, 1000, 0, 0 );
}

void Detector::drawContours(cv::Mat &image, vector<vector<cv::Point> > contours)
{
    cv::drawContours( image, contours, -1, cv::Scalar(0, 255, 0), 1 );	
}

////mahdi below was for brightening the image, not used now
//void FindContoursDetector::detectFeatures(cv::Mat& image, cv::Mat& original_image, vector<ParticleImage::Ptr>& features, vector<vector<cv::Point>>& contours, int index)
//{
//	// Brighten the image by adding a value (e.g., 50) to each pixel
//	image += cv::Scalar((int)params.min_contour_area);
//	original_image += cv::Scalar((int)params.min_contour_area);
//	//cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
//	//cv::normalize(original_image, original_image, 0, 255, cv::NORM_MINMAX);
//}

void FindContoursDetector::detectFeatures(cv::Mat& image, cv::Mat& original_image, vector<ParticleImage::Ptr>& features, vector<vector<cv::Point>>& contours, int index)
{
	double tempx, tempy;
	int temp_xx, temp_yy;
	int height = image.rows;
	int width = image.cols;

	float temp_radius;
	double intensity = -1;
	double max_intensity = -1;
	double mean_intensity = 0;
	double area;
	double roundness;
	double temp_perimeter;

	// temp radius

	double pi = 3.141592653;

	int moving_radius = 1;
	int count = 0;

    cv::findContours(image, contours, params.mode, params.method);

    for(int c = 0; c < contours.size(); ++c) {
        if( contours[c].size() > (int)params.min_contour_area  && contours[c].size() < (int)params.max_contour_area) {
			area = (double)contours[c].size();
			
			cv::Moments mom = cv::moments(cv::Mat(contours[c]) );
            if( mom.m00 > 0 ) {
				tempx = mom.m10 / mom.m00;   // here x is width or height? specify later
				tempy = mom.m01 / mom.m00;
				
				// find the temp radius of min enclousing circle of contour
				cv::minEnclosingCircle(contours[c], cv::Point2f(), temp_radius);

				// find the circularity
				temp_perimeter = (double)cv::arcLength(contours[c], true);
				roundness = 4 * pi * (area / (temp_perimeter * temp_perimeter));

				//// find the max intensity and mean intensity
				//max_intensity = -1.0;
				//mean_intensity = 0.0;
				//temp_xx = (int)tempx;
				//temp_yy = (int)tempy;
				//count = 0;

				//double temp_intensity = 0.0;
				//int x_max = -1;
				//int y_max = -1;

				//for (int i = temp_xx - moving_radius; i <= temp_xx + moving_radius; i++) {
				//	for (int j = temp_yy - moving_radius; j <= temp_yy + moving_radius; j++) {
				//		if (i >= 0 && i < width && j >= 0 && j < height) {
				//			temp_intensity = (double)original_image.at<uchar>(j, i);

				//			if (i == temp_xx && j == temp_yy) {
				//				intensity = temp_intensity;
				//			}

				//			if (temp_intensity > max_intensity) {
				//				max_intensity = temp_intensity;
				//				x_max = i;
				//				y_max = j;
				//			}

				//			mean_intensity += temp_intensity;
				//			count++;
				//		}
				//	}
				//}

				//mean_intensity = mean_intensity / (double)count;

				//features.push_back( ParticleImage::create(static_cast<int>(features.size()), tempx, tempy, (double)temp_radius) );
				//features.push_back(ParticleImage::create(static_cast<int>(features.size()), tempx, tempy, (double)temp_radius,
				//	intensity, max_intensity, mean_intensity,
				//	area, roundness, temp_perimeter,
				//	(int)(x_max - temp_xx), (int)(y_max - temp_yy)));

				features.push_back(ParticleImage::create(static_cast<int>(features.size()), tempx, tempy, (double)temp_radius, area, roundness));

			}
        }
    }
}

FindContoursDetector::~FindContoursDetector()
{
    cout << "Find Contours detector destructed" << endl;
}

bool FindContoursDetector::isGfftDetector() {
	return is_gfft;
}

//GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector()
//{
//	is_gfft = true;
//    cout << "Good Features To Track (GFTT) detector constructed" << endl;
//}
//
//void GoodFeaturesToTrackDetector::initCudaVecs(int numCameras, cv::Mat full_img_type, cv::Mat half_img_type) {
//	if(numCameras){
//		for (int i = 0; i < numCameras; i++) {
//			//Define detectors
//
//			//Initialize the Pinned Memory with input image
//			cv::Mat full_img = cv::Mat::zeros(full_img_type.rows, full_img_type.cols, CV_8UC1);
//			cv::Mat half_img = cv::Mat::zeros(half_img_type.rows, half_img_type.cols, CV_8UC1);
//			cv::cuda::HostMem fullSrcHostMem(full_img, cv::cuda::HostMem::PAGE_LOCKED);
//			cv::cuda::HostMem halfSrcHostMem(half_img, cv::cuda::HostMem::PAGE_LOCKED);
//
//			hostMemSrcImageVec.push_back(halfSrcHostMem);
//			//hostMemSrcImageVec.push_back(fullSrcHostMem);
//
//			hostMemDstImageVec.push_back(fullSrcHostMem);
//			hostMemThresholdImageVec.push_back(fullSrcHostMem);
//
//			//Initialize the output Pinned Memory with reference to output Mat
//			cv::Mat zero_type_img_corners = cv::Mat::zeros(1, params.max_corners, CV_32FC2);
//			cv::cuda::HostMem srcDstMem(cv::cuda::HostMem(zero_type_img_corners, cv::cuda::HostMem::PAGE_LOCKED));
//			cpuCentroidsVec.push_back(zero_type_img_corners);			// initialize cpuCentroidsVec
//			hostMemCornersVec.push_back(srcDstMem);
//			hostMemCentroidsVec.push_back(srcDstMem);			// initialize hostCentroidsVec
//
//			cv::Ptr<cv::cuda::CornersDetector> ptr = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, params.max_corners,
//				params.quality_level, params.min_distance, params.neighborhood_size, params.use_harris, params.k);
//			cvCornerDetectorVec.push_back(ptr);
//
//			//Define GPU Mats
//			gpuSrcImageVec.push_back(cv::cuda::GpuMat());
//			gpuDstImageVec.push_back(cv::cuda::GpuMat());
//			gpuThresholdImageVec.push_back(cv::cuda::GpuMat());
//			gpuDstCurrentCornersVec.push_back(cv::cuda::GpuMat());
//			gpuPrevCornersVec.push_back(cv::cuda::GpuMat());
//			gpuPrevCentroidsVec.push_back(cv::cuda::GpuMat());
//			gpuCentroidsVec.push_back(cv::cuda::GpuMat());			// initialize gpuCentroidsVec
//			gpuThresholdImageVec.push_back(cv::cuda::GpuMat());
//
//			//Define vector to store corners in CPU // TODO INITIALIZE HERE?
//			cpuDstImageVec.push_back(cv::Mat());
//			cpuCornersVec.push_back(cv::Mat());
//			cpuThresholdImageVec.push_back(cv::Mat());
//
//			//Initialize the cuda stream
//			gfftStreamsVec.push_back(cv::cuda::Stream());
//
//			// initialize gpuFilterVec
//			cv::cuda::GpuMat filter(1, params.max_corners, CV_8UC1);
//			gpuCornersFilterVec.push_back(filter);
//			gpuCentroidsFilterVec.push_back(filter);			// initialize gpuCentroidsFilterVec
//
//			// initialize hostFilterVec
//			cv::cuda::HostMem hostFilter;
//			hostFilter = cv::cuda::HostMem(1, params.max_corners, cv::cuda::HostMem::PAGE_LOCKED);
//			hostMemFilterVec.push_back(hostFilter);
//			hostMemCentroidsFilterVec.push_back(hostFilter);			// initialize hostCentroidsFilterVec
//
//			// initialize cpuFilterVec
//			cv::Mat cpuFilter = cv::Mat::zeros(1, params.max_corners, CV_8UC1);
//			cpuCornersFilterVec.push_back(cpuFilter);
//			cpuCentroidsFilterVec.push_back(cpuFilter);			// initialize cpuCentroidsFilterVec
//
//			// initiaze prevCornersNum and prevCentroidsNum
//			prevCornersNum.push_back(-1);
//			prevCentroidsNum.push_back(-1);
//		}
//
//		cout << "-----------Succeeed to init gfft vects ------------" << endl;
//	}
//	
//	m_threshold = 100;
//	m_max_threshold = 255;
//
//	for (int i = 0; i < numCameras; i++) {
//		thresholds.push_back(m_threshold);
//	}
//
//	//currentIdx = 0;
//
//	//cv::createTrackbar("currentIdxGpu", string(), &currentIdx, numCameras-1, 0, 0);
//	cv::createTrackbar("GpuThresholdImage0", string(), &thresholds[0], m_max_threshold, 0, 0);
//	cv::createTrackbar("GpuThresholdImage1", string(), &thresholds[1], m_max_threshold, 0, 0);
//	cv::createTrackbar("GpuThresholdImage2", string(), &thresholds[2], m_max_threshold, 0, 0);
//	cv::createTrackbar("GpuThresholdImage3", string(), &thresholds[3], m_max_threshold, 0, 0);
//	cv::createTrackbar("GpuThresholdImage4", string(), &thresholds[4], m_max_threshold, 0, 0);
//	cv::createTrackbar("GpuThresholdImage5", string(), &thresholds[5], m_max_threshold, 0, 0);
//
//	cout << "\t\tAdded Gpu Threshold process to Control Window" << endl;
//}

//void GoodFeaturesToTrackDetector::detectFeatures( cv::Mat &image, cv::Mat& original_image, vector<ParticleImage::Ptr> &features, vector<vector<cv::Point>>& contours, int index)
//{
//    //CV_Assert(image.depth() != sizeof(uchar));
//    if (! image.isContinuous())
//        cout << "Mat is not continuous "<< image.size().width << " " << image.size().height << endl;
//
//	// start debugging
//	//cv::resize(image, image, cv::Size(), 2, 2, cv::INTER_LINEAR);
//
//	image.copyTo(hostMemSrcImageVec[index].createMatHeader());
//
//	gpuSrcImageVec[index].upload(hostMemSrcImageVec[index], gfftStreamsVec[index]);
//
//	cv::cuda::resize(gpuSrcImageVec[index], gpuDstImageVec[index], cv::Size(), 2, 2, cv::INTER_LINEAR, gfftStreamsVec[index]);
//
//	cv::cuda::threshold(gpuDstImageVec[index], gpuThresholdImageVec[index], thresholds[index], m_max_threshold, 0, gfftStreamsVec[index]);
//	//cv::cuda::threshold(gpuDstImageVec[index], gpuThresholdImageVec[index], m_threshold, m_max_threshold, 0, gfftStreamsVec[index]);  // This line is for unifrom threshold
//
//	gpuThresholdImageVec[index].download(hostMemThresholdImageVec[index], gfftStreamsVec[index]);
//
//	cv::resize(image, image, cv::Size(), 2, 2, cv::INTER_LINEAR);
//	//gpuDstImageVec[index].download(hostMemDstImageVec[index], gfftStreamsVec[index]);
//	//image.resize(image.rows, image.cols);
//
//	//gfftStreamsVec[index].waitForCompletion();
//	//cpuDstImageVec[index] = hostMemDstImageVec[index].createMatHeader();
//
//	//cpuDstImageVec[index].copyTo(image);
//
//	cpuThresholdImageVec[index] = hostMemThresholdImageVec[index].createMatHeader();
//
//	cvCornerDetectorVec[index]->detect(gpuThresholdImageVec[index], gpuDstCurrentCornersVec[index], cv::cuda::GpuMat(), gfftStreamsVec[index]);
//
//	// end debugging
//
//
//	//// start release
//
//	//image.copyTo(hostMemSrcImageVec[index].createMatHeader());
//
//	//gpuSrcImageVec[index].upload(hostMemSrcImageVec[index], gfftStreamsVec[index]);
//
//	//cv::cuda::resize(gpuSrcImageVec[index], gpuDstImageVec[index], cv::Size(), 2, 2, cv::INTER_LINEAR, gfftStreamsVec[index]);
//
//	//gpuDstImageVec[index].download(hostMemDstImageVec[index], gfftStreamsVec[index]);
//
//	//gfftStreamsVec[index].waitForCompletion();
//
//	//image = hostMemDstImageVec[index].createMatHeader();
//
//	//// perform the threshold
//	//cv::cuda::threshold(gpuDstImageVec[index], gpuThresholdImageVec[index], m_threshold, m_max_threshold, 0, gfftStreamsVec[index]);
//
//	//gpuThresholdImageVec[index].download(hostMemThresholdImageVec[index], gfftStreamsVec[index]);
//
//	//gfftStreamsVec[index].waitForCompletion();
//
//	//cpuThresholdImageVec[index] = hostMemThresholdImageVec[index].createMatHeader();
//
//	//// perform the gfft
//	//cvCornerDetectorVec[index]->detect(gpuThresholdImageVec[index], gpuDstCurrentCornersVec[index], cv::cuda::GpuMat(), gfftStreamsVec[index]);
//
//	//// end release
//
//	///////////////////////////////////////////////////////////////
//	double tempx, tempy;
//
//	cv::findContours(cpuThresholdImageVec[index], contours, params.mode, params.method);
//
//	int non_zero_centroid_count = 0;
//
//	for (int c = 0; c < contours.size(); ++c) {
//		if (contours[c].size() >(double)params.min_contour_area  && contours[c].size() < (double)params.max_contour_area) {
//			cv::Moments mom = cv::moments(cv::Mat(contours[c]));
//			if (mom.m00 > 0) {
//				tempx = mom.m10 / mom.m00;   // here x is width or height? specify later
//				tempy = mom.m01 / mom.m00;
//
//				if (non_zero_centroid_count < cpuCentroidsVec[index].cols) {
//					cpuCentroidsVec[index].at<float2>(0, non_zero_centroid_count) = make_float2(tempx, tempy);
//					non_zero_centroid_count++;
//				}
//			}
//		}
//	}
//
//	cpuCentroidsVec[index].copyTo(hostMemCentroidsVec[index].createMatHeader());
//	gpuCentroidsVec[index].upload(cpuCentroidsVec[index], gfftStreamsVec[index]);  // TODO??? shouldn't upload the hostMem??
//
//	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//	gfftStreamsVec[index].waitForCompletion();
//	
//	cudaStream_t stream_ = cv::cuda::StreamAccessor::getStream(gfftStreamsVec[index]);
//
//	// download result to output pinned memory
//	gpuDstCurrentCornersVec[index].download(hostMemCornersVec[index], gfftStreamsVec[index]);
//
//	int numCorners = gpuDstCurrentCornersVec[index].cols;
//
//	if (numCorners && prevCornersNum[index] > 0 && prevCentroidsNum[index] > 0) {
//		int size_1 = numCorners;
//		int size_2 = prevCornersNum[index];
//		int size_3 = prevCentroidsNum[index];
//
//		int max_size = max<int>(size_2, size_3);
//
//		float* distances_1 = NULL;
//		gpuErrchk(cudaMalloc((void**)(&distances_1), size_1 * size_2 * sizeof(float)));
//
//		float* distances_2 = NULL;
//		gpuErrchk(cudaMalloc((void**)(&distances_2), size_1 * size_3 * sizeof(float)));
//
//		int width = (int)(max_size / 32 + 1);
//		int height = (int)(size_1 / 32 + 1);
//
//		dim3 dimBlock(32, 32, 1);
//		dim3 dimGrid(width, height, 1);
//
//		float2* currentCornersPtr = gpuDstCurrentCornersVec[index].ptr<float2>();
//		float2* previousCornersPtr = gpuPrevCornersVec[index].ptr<float2>();
//		float2* previousCentroidsPtr = gpuPrevCentroidsVec[index].ptr<float2>();
//
//		bool* boolVecPtr = gpuCornersFilterVec[index].ptr<bool>();
//
//		//calculateDistances << <dimGrid, dimBlock, 0, stream_ >> > (currentCornersPtr, previousCornersPtr, size_1, size_2, distances, 10, boolVecPtr);
//		calculateDistances_centroids << <dimGrid, dimBlock, 0, stream_ >> > (currentCornersPtr, previousCornersPtr, previousCentroidsPtr, size_1, size_2, size_3, distances_1, distances_2, 20, boolVecPtr);
//
//		gpuErrchk(cudaPeekAtLastError());
//		gpuErrchk(cudaFree(distances_1));
//		gpuErrchk(cudaFree(distances_2));
//
//		currentCornersPtr = NULL;
//		previousCornersPtr = NULL;
//		previousCentroidsPtr = NULL;
//		boolVecPtr = NULL;
//	}
//	else {
//		// invoke the test kernel
//		dim3 dimBlock(32, 1, 1);
//		dim3 dimGrid((int)(params.max_corners / 32 + 1), 1, 1);
//		test_kernel << <dimGrid, dimBlock, 0, stream_ >> > (gpuCornersFilterVec[index].ptr<bool>(), numCorners);
//	}
//
//	gpuErrchk(cudaPeekAtLastError());
//
//	gfftStreamsVec[index].waitForCompletion();
//
//	gpuCornersFilterVec[index].download(hostMemFilterVec[index], gfftStreamsVec[index]);  // here do we need to wait for the kernel to finish?
//	
//	gfftStreamsVec[index].waitForCompletion();
//
//	// obtain data back go cpu memory  // TODO Before was in fornt of stream blocking
//	cpuCornersVec[index] = hostMemCornersVec[index].createMatHeader();
//	cpuCornersFilterVec[index] = hostMemFilterVec[index].createMatHeader();
//
//	if (cpuCornersVec[index].cols) {
//		cv::Point2f pt;
//
//		for (int i = 0; i < cpuCornersVec[index].cols; i++) {
//			if (i < params.max_corners && cpuCornersFilterVec[index].at<bool>(0, i)) {
//				pt = cpuCornersVec[index].at<cv::Point2f>(0, i);
//				features.push_back(ParticleImage::create(features.size(), pt.x, pt.y, 5));
//			}
//		}
//	}
//
//	//// push the centroids to features vector
//	//for (int i = 0; i < non_zero_centroid_count; i++) {
//	//	features.push_back(ParticleImage::create(features.size(), cpuCentroidsVec[index].at<float2>(0, i).x, cpuCentroidsVec[index].at<float2>(0, i).y, 8));
//	//}
//
//	gpuPrevCornersVec[index] = gpuDstCurrentCornersVec[index];
//	prevCornersNum[index] = numCorners;
//	gpuPrevCentroidsVec[index] = gpuCentroidsVec[index];
//	prevCentroidsNum[index] = non_zero_centroid_count;
//	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//}
//
//void GoodFeaturesToTrackDetector::addControls()
//{
//	cout << "\t--Adding Find Contours Detector Controls to window: " << endl;
//    cout << "GoodFeaturesToTrack detector controls added" << endl;
//	cv::createTrackbar("Min Area", string(), &params.min_contour_area, 500, 0, 0);
//	cv::createTrackbar("Max Area", string(), &params.max_contour_area, 1000, 0, 0);
//	cv::createTrackbar("Max Corners", string(), &params.max_corners, 200, 0, 0);
//}
//
//GoodFeaturesToTrackDetector::~GoodFeaturesToTrackDetector()
//{
//    cout << "GoodFeaturesToTrack detector destructed" << endl;
//}
//
//bool GoodFeaturesToTrackDetector::isGfftDetector() {
//	return is_gfft;
//}
//
//// what is this function for? couldn't find any call in dataaquisition.cpp
//void processImages( Camera& camera, ImageProcessor& processor, Detector& detector )
//{
//	cout << "Camera "<< camera.id << ": --Processing Images" << endl;
//    	
//	stringstream result_window;
//	result_window << camera.name << ": detected particles";
//	
//	processor.addControls();
//	detector.addControls();
//	cv::waitKey(10);
//
//	size_t number_of_frames = camera.imagelist.size();
//	camera.frames.resize( number_of_frames );
//	for (int i = 0; i < number_of_frames; ++i) {
//		cv::Mat temp_image = camera.frames[i].image.clone();
//		processor.processImage( temp_image );
//		cv::imshow("processed image", temp_image );
//		detector.detectFeatures( temp_image, temp_image, camera.frames[i].particles, camera.frames[i].contours, i);
//		detector.drawResult( camera.frames[i] );
//		cv::imshow(result_window.str(), camera.frames[i].image);
//	}
//	cv::destroyWindow( result_window.str() );
//}

void testDetectedParticles(
		vector<ParticleImage::Ptr>& true_particles,
		vector<ParticleImage::Ptr>& detected_particles,
		double& accuracy, double& meanError)
{
	double total_residual = 0;
	int total_matches = 0;
	for (int m = 0; m < true_particles.size(); m++) {
		int number_of_matches = 0;
		vector<double> residuals(true_particles.size(), 0);
		for(int n = 0; n < detected_particles.size(); n++) {
			double diff_x = true_particles[m]->x - detected_particles[n]->x;
			double diff_y = true_particles[m]->y - detected_particles[n]->y;
			double r = sqrt( diff_x * diff_x + diff_y * diff_y );
			//if (r <= 4) {   //FIXME: make this a function of actual particle radius
            if(r <= true_particles[m]->radius) {   //FIXME: make this a function of actual particle radius
			//if (r <= 1.0f) {   //FIXME: make this a function of actual particle radius

				residuals[m] += r;
				number_of_matches++;
				break;                                    //FIXME: consider if more than one particle matches
			}
		}
		total_matches += number_of_matches;
		total_residual += residuals[m];
	}

	accuracy = (double)total_matches / true_particles.size();
	meanError = (total_matches > 0 ? total_residual / total_matches : 0);
	//TODO: print the results to a file and plot residuals (histogram)
	cout << "Correct Ratio: " << accuracy;
	cout << "\tCover Ratio: " << (double)total_matches / true_particles.size();
	cout << "\tAvg Frame Residual: "
		<< meanError << endl;
	//cout << "\tCover Ratio: " << (double)total_matches / true_particles.size();
	//cout << "\tAvg Frame Residual: "
	//		<< (total_matches > 0 ? total_residual / total_matches : 0) << endl;


}

} /*NAMESPACE_PT_*/

