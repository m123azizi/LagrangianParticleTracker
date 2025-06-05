/*
Data aquisition module headers:
These classes provide data aquisition funcionality to the particle tracking system and the overall multi-threaded streaming framework.

copyright: Douglas Barker 2011
*/

#ifndef DATAAQUSITION_H_
#define DATAAQUSITION_H_

//#include <filesystem> //mahdi for cheking folders exists when writing raw dat. update: required cpp standard ver 17 or above, incompatible with the project
#include <core.hpp>
#include <calib.hpp>
#include <datagen.hpp>
#include <imageproc.hpp>
#include <correspond.hpp>

#include <string>//new added by yu

#ifdef USE_CUDA
#include <correspondcuda.h>
#endif

#include <tracking.hpp>
#include <visualization.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/chrono.hpp>
// mahdi below included for time stamp of writing output data
#include <iomanip>
#include <ctime>
#include <chrono>
#include <sstream> 
#include <regex> // mahdi at 5/13/2024 for finding patterns in cumulative text files of input data
#include <unordered_map> //for lisintg files in a folder for calibration read back
#include <boost/filesystem.hpp> //for lisintg files in a folder for calibration read back
//end mahdi
#ifdef USE_NP_CAMERASDK
#include <windows.h>
#include "cameralibrary.h"
#endif


namespace lpt {

	using namespace std;
	using namespace CameraLibrary;

	class Video;
	class Recorder;
	class CameraSystem;
	class VirtualCameras;
	class VirtualCamerasFile; //mahdi
	class VirtualCamerasImg; //mahdi
	class SIST_VirtualCamerasFile; //mahdi
	class NICT_VirtualCamerasFile; //mahdi
	class SINTC_VirtualCamerasFile; //mahdi
	class Optitrack;
	class StreamingPipeline;

	/*
	 * This class provides the video object that the Recorder class will use to store images and convert into avi files
	 */
	class Video {
	public:
		typedef std::shared_ptr<Video> Ptr;
		static inline Video::Ptr create(string path = "./") { return Video::Ptr(new Video(path)); }

		vector<cv::Mat> video_frames;
		Video(string path = "./") : file_path(path) {}
		inline void setFilePath(string path) { file_path = path; }
		inline void addImageToVideo(cv::Mat image) { video_frames.push_back(image); }
		inline size_t getVideoLength() const { return video_frames.size(); }
		void writeAVI(int playback_fps, int codec);
		void writeImages(string basename, string filetype = ".jpg");
		void readAVI();
		void readImages(string basename, int numframes, string filetype = ".jpg");

		virtual ~Video() {}

	private:
		string file_path;
	};

	/*
	 * This class provides the functionality to record avi files from the camera system images
	 */
	class Recorder {
	public:
		typedef std::shared_ptr<lpt::Recorder> Ptr;
		static inline lpt::Recorder::Ptr create(int num_cams, string path = "./", int length = 900) {
			return lpt::Recorder::Ptr(new lpt::Recorder(num_cams, path, length));
		}

		Recorder(int num_cams, string path = "./", int length = 900)
			: number_of_cameras(num_cams), path(path), clip_length(length)
		{
			videos.resize(number_of_cameras);
			imagelists.resize(number_of_cameras);
			snapshot_requested = false;
			record_video = false;
			snapshot_id = -1;
			video_id = -1;
		}

		virtual ~Recorder() {
			if (getVideoCount() > 0) {
				cout << "Writing " << getVideoCount() << " video sequence to file for each camera" << endl;
				int playback_rate;
				cout << "Enter desired playback frame rate" << endl;
				cin >> playback_rate;
				writeVideosToFile(playback_rate);
			}
			if (getSnapShotCount() > 0)
				writeSnapShotImageLists();
		}

		inline void setClipLength(int length) { clip_length = length; }
		inline void setNumCameras(int num) {
			number_of_cameras = num;
			videos.resize(number_of_cameras);
			imagelists.resize(number_of_cameras);
		}
		inline void requestSnapShot() { snapshot_requested = true; }
		inline bool isSnapShotRequested() const { return snapshot_requested; }
		inline bool isVideoRecording() const { return record_video; }
		inline size_t getVideoCount() const { return videos[0].size(); }
		inline size_t getSnapShotCount() const { return imagelists[0].size(); }
		void takeSnapShot(vector<cv::Mat>& frames);
		void writeSnapShotImageLists();
		void createVideos();
		void addFramesToVideos(vector<cv::Mat>& frames);
		void writeVideosToFile(int playback_fps = 10, int codec = -1);

	private:
		bool snapshot_requested;
		bool record_video;
		int video_id;
		int snapshot_id;
		int clip_length;
		int number_of_cameras;
		vector<vector<string> > imagelists;
		vector<vector<Video> > videos;
		string path;

	};

	/*
	 * This is a generic base class for the camera system used in particle tracking
	 */
	class CameraSystem {
	public:
		typedef std::shared_ptr < lpt::CameraSystem > Ptr;

		CameraSystem() : collect_particledata_from_camera(true) { }
		// some functions are pure virtual =0, and this class is abstract, so it cannot be instantiated directly, it is only a base
		
		virtual void addControls() = 0;
		virtual bool initializeCameras() = 0;
		virtual bool grabFrameGroup(lpt::ImageFrameGroup& frame_group) = 0;
		virtual void shutdown() = 0;
		virtual ~CameraSystem() {}
		bool initialize();

		inline void setCurrentCameraIndex(int& idx) { current_camera_idx = &idx; }// for optitrack only to make the camera index changeable to set different threshold
		void loadCameraParams(const string filepath);
		void loadCameraPairParams(const string filepath);

		inline shared_ptr < lpt::SharedObjects > getSharedObjects() { return shared_objects; }
		inline void setSharedObjects(shared_ptr < lpt::SharedObjects > new_shared_objects) { shared_objects = new_shared_objects; }

		inline void setInputDataPath(string& path) { shared_objects->input_path = path; }
		inline string getInputDataPath() { return shared_objects->input_path; }
		inline void setOutputDataPath(string& path) { shared_objects->output_path = path; }
		inline string getOutputDataPath() { return shared_objects->output_path; }

		inline void setImageType(cv::Mat image_type) { shared_objects->image_type = image_type; }
		inline cv::Mat getImageType() { return shared_objects->image_type; }

		inline void setParticleCollectionFromCamera(bool state) { collect_particledata_from_camera = state; }
		inline bool getParticleCollectionFromCamera() { return collect_particledata_from_camera; }
		inline string getWindowName() { return window_name; }
		inline bool areCamerasRunning() const { return cameras_status; }
		inline void setCamerasStatus(bool state) { cameras_status = state; cout << "Cameras " << (state ? "on" : "off") << endl; }
		inline int getFrameRate() { return shared_objects->frame_rate; }
		inline void setCalibrator(std::shared_ptr<lpt::Calibrator> calibrator) { this->calibrator = calibrator; }
		bool WriteRawDataRequested = false;

		//inline bool getCurParCol(int idx) { return collect_particledata_from_camera_vec[idx]; }
		//inline void setCurCamParCol(bool state) { collect_particledata_from_camera_vec[current_camera_idx] = state; }
	protected:
		shared_ptr < lpt::SharedObjects > shared_objects;
		shared_ptr < lpt::Calibrator > calibrator; //access setframeinterval of calibrator
		bool cameras_status;
		bool collect_particledata_from_camera;
		string window_name;
		int* current_camera_idx;

		//vector<bool> collect_particledata_from_camera_vec; // this vector to store the seperate switch indicates which camera collect particle from camera		
	};
	/* created by mahdi
	 * This class provides virtual cameras to allow simulation of particle images based on loading images and particles from files, saved from real simulations
	 */
	class VirtualCamerasFile : public CameraSystem { // base virtual class for virtual cameras from file system
	public:
		typedef std::shared_ptr<lpt::VirtualCamerasFile> Ptr;
		VirtualCamerasFile(const std::string& camerasfile, const std::string& image_dir,
			const std::string& text_dir, int frame_start, int frame_end,
			int frame_step, int fps)
			: camerasfile(camerasfile), image_dir(image_dir), text_dir(text_dir),
			frame_start(frame_start), frame_end(frame_end), frame_index(frame_start),
			frame_step(frame_step), virtual_fps(fps), frame_rate_level(8000000) {
			window_name = "Virtual Camera from File Control Window";
			std::cout << "Created Virtual Camera File System" << std::endl;
		}
		virtual ~VirtualCamerasFile() = default;

		virtual bool initializeCameras() = 0;
		virtual void addControls();
		virtual bool grabFrameGroup(lpt::ImageFrameGroup& frame_group) = 0;
		void shutdown() = 0;
		void VFcreateImage(lpt::ImageFrame& frame); // for creating images from text file on the fly, not reading from file
		//inline lpt::DataSetGeneratorFile::Ptr getGeneratorFile() { return generatorFile; }
	protected:
		//lpt::DataSetGeneratorFile::Ptr generatorFile;
		int frame_start, frame_end, frame_index, virtual_fps, frame_step, frame_rate_level;
		std::string camerasfile, image_dir, text_dir;		
	};

	class SIST_VirtualCamerasFile : public VirtualCamerasFile { //separate image and separate text files
	public:
		typedef std::shared_ptr<lpt::SIST_VirtualCamerasFile> Ptr;
		static SIST_VirtualCamerasFile::Ptr create(const string& camerasfile, const string& image_dir,
			const string& text_dir, int frame_start, int frame_end, int frame_step, int fps) {
			return std::make_shared<SIST_VirtualCamerasFile>(camerasfile, image_dir, text_dir, frame_start, frame_end, frame_step, fps);
		}

		SIST_VirtualCamerasFile(const std::string& camerasfile, const std::string& image_dir,
			const std::string& text_dir, int frame_start, int frame_end, int frame_step, int fps) :
			VirtualCamerasFile(camerasfile, image_dir, text_dir, frame_start, frame_end, frame_step, fps) {}
		bool initializeCameras() override;
		bool grabFrameGroup(lpt::ImageFrameGroup& frame_group) override;
		void shutdown() override;
	};

	class NICT_VirtualCamerasFile : public VirtualCamerasFile { //no image, combined text files
	public:
		typedef std::shared_ptr<lpt::NICT_VirtualCamerasFile> Ptr;
		static NICT_VirtualCamerasFile::Ptr create(const string& camerasfile, const string& image_dir,
			const string& text_dir, int frame_start, int frame_end, int frame_step, int fps) {
			return std::make_shared<NICT_VirtualCamerasFile>(camerasfile, image_dir, text_dir, frame_start, frame_end, frame_step, fps);
		}

		NICT_VirtualCamerasFile(const std::string& camerasfile, const std::string& image_dir,
			const std::string& text_dir, int frame_start, int frame_end, int frame_step, int fps) :
			VirtualCamerasFile(camerasfile, image_dir, text_dir, frame_start, frame_end, frame_step, fps) {
			frame_start_pattern = std::regex(R"(==== p_(\d+)_(\d+)\.txt ====)");
		}
		bool initializeCameras() override;
		bool grabFrameGroup(lpt::ImageFrameGroup& frame_group) override;
		void shutdown() override;
	protected:
		std::vector<std::ifstream> input_txt_files;
		std::vector<std::streampos> input_txt_files_last_positions;
		std::regex frame_start_pattern;
	};

	class SINTC_VirtualCamerasFile : public VirtualCamerasFile { //separate image, no text, for calibration files
	public:
		typedef std::shared_ptr<lpt::SINTC_VirtualCamerasFile> Ptr;
		static SINTC_VirtualCamerasFile::Ptr create(const string& camerasfile, const string& image_dir,
			const string& text_dir, int frame_start, int frame_end, int frame_step, int fps) {
			return std::make_shared<SINTC_VirtualCamerasFile>(camerasfile, image_dir, text_dir, frame_start, frame_end, frame_step, fps);
		}

		SINTC_VirtualCamerasFile(const std::string& camerasfile, const std::string& image_dir,
			const std::string& text_dir, int frame_start, int frame_end, int frame_step, int fps) :
			VirtualCamerasFile(camerasfile, image_dir, text_dir, frame_start, frame_end, frame_step, fps) {}
		bool initializeCameras() override;
		bool grabFrameGroup(lpt::ImageFrameGroup& frame_group) override;
		void shutdown() override;
	private:
		int blank_repeat_count = 4;
		int current_blank_count = 0;
		unordered_map<int, vector < string>> file_map;
		unordered_map<int, int> file_index_map; // Map to store the current file index for each camera
		bool calibrator_settings = false;
	};

	class VirtualCamerasImg : public CameraSystem { // base virtual class for virtual cameras from image files
	public:
		typedef std::shared_ptr<lpt::VirtualCamerasImg> Ptr;
		VirtualCamerasImg(const std::string& camerasfile, const std::string& image_dir,
			int frame_start, int frame_end, int frame_step, int fps) : 
			camerasfile(camerasfile), image_dir(image_dir),
			frame_start(frame_start), frame_end(frame_end), frame_index(frame_start),
			frame_step(frame_step), virtual_fps(fps), frame_rate_level(4000000) {
			window_name = "Virtual Camera from Image Control Window";
			std::cout << "Created Virtual Camera Image System" << std::endl;
		}
		virtual ~VirtualCamerasImg() = default;

		virtual bool initializeCameras();
		virtual void addControls();
		virtual bool grabFrameGroup(lpt::ImageFrameGroup& frame_group);
		void shutdown();		
	private:
		int frame_start, frame_end, frame_index, virtual_fps, frame_step, frame_rate_level;
		std::string camerasfile, image_dir;

		//
		int blank_repeat_count = 4;
		int current_blank_count = 0;
		//unordered_map<int, vector < string>> file_map;
		//unordered_map<int, int> file_index_map; // Map to store the current file index for each camera		
	};



	//end by mahdi

	/*
	 * This class provides virtual cameras to allow simulation of particle images based on synthetic trajectories given in a file
	 */
	class VirtualCameras : public CameraSystem {
	public:
		typedef std::shared_ptr<VirtualCameras> Ptr;
		static VirtualCameras::Ptr create(string cams_file, string traj_file) {
			return std::make_shared<VirtualCameras>(cams_file, traj_file);
		}

		VirtualCameras() {}

		VirtualCameras(string cams_file, string traj_file) : cameras_file(cams_file), trajectory_file(traj_file), frame_rate_level(10000), frame_index(0) {
			this->generator = std::make_shared<lpt::DataSetGenerator>();
			window_name = "Virtual Camera Control Window";
			cout << "Created Virtual Camera System" << endl; //some comment
		}

		virtual bool initializeCameras();
		virtual void addControls();
		virtual bool grabFrameGroup(lpt::ImageFrameGroup& frame_group);
		void shutdown();
		inline lpt::DataSetGenerator::Ptr getGenerator() { return generator; }

		int frame_rate_level;
	private:
		string cameras_file;
		string trajectory_file;
		lpt::DataSetGenerator::Ptr generator;
		int frame_index;
	};

#ifdef USE_NP_CAMERASDK

	/*
	 * This class identifies, controls, and grabs images/data from Optitrack motion capture cameras using the Natural Point C++ camera SDK
	 */
	class Optitrack : public CameraSystem {
	public:
		typedef std::shared_ptr<Optitrack> Ptr;
		static Optitrack::Ptr create() { return make_shared< lpt::Optitrack >(); }
		Optitrack() : frame_count(0)
		{
			window_name = "Optitrack Control Window";
			//initializeWriterThreadsAndQueues is called under Optitrack::initializeCameras()
		}

		bool initializeCameras();
		void addControls();
		bool grabFrameGroup(lpt::ImageFrameGroup& frame_group);
		void shutdown();
		//mahdi all below
		//bool writePGM(const std::string& filename, const cv::Mat& image);//mahdi commneted on 5/6/2024
		void processCameraData(int camera_id, const shared_ptr<const CameraLibrary::FrameGroup>& native_frame_group, lpt::ImageFrameGroup& frame_group); //mahdi
		void writeTxtFileGeneral(int cameraIndex);
		void writeImgFileGeneral(int cameraIndex);
		//void writeTxtFiles();
		void startCameraTxtWriters();
		void stopCameraTxtWriters();
		void startCameraImgWriters();
		void stopCameraImgWriters();
		
		// start of writing multithreaded txt files
		//void writeTxtFiles0();
		//void writeTxtFiles1();
		//void writeTxtFiles2();
		//void writeTxtFiles3();
		//void writeTxtFiles4();
		//void writeTxtFiles5();
		//void writeTxtFiles6();
		//void writeTxtFiles7();
		// end of writing multithreaded txt files
		vector<shared_ptr<CameraLibrary::Camera>>& getOptitrackCameras() { return optitrack_cameras; }

		/*----------------------------------*/
		// added by yu
		//oduleSync* getSyncModule() { return sync; }
		vector<shared_ptr<const CameraLibrary::Frame>>& getOptitrackFrames() { return optitrack_frames; }
		/*----------------------------------*/

		//----CALL BACK FUNCTIONS-----
		friend void callbackSetVideoType(int mode, void* data);
		friend void callbackSetThreshold(int value, void* data);
		friend void callbackSetIRFilter(int state, void* data);
		friend void callbackSetAEC(int state, void* data);
		friend void callbackSetAGC(int state, void* data);
		friend void callbackSetTextOverlay(int state, void* data);
		friend void callbackSetExposure(int value, void* data);
		friend void callbackSetIntensity(int value, void* data);
		friend void callbackSetFrameRate(int value, void* data);
		friend void callbackSetWritingRawData(int state, void* data);//mahdi for raw data write
		friend void callbackSetCurThreshold(int value, void* data);
		friend void callbackSetCurExposure(int value, void* data);
		//friend void callbackSetCurrentVideoType(int mode, void* data);
		//friend void callbackSetCurrentExpTime(int value, void* data);

	private:
		int txt_queue_capacity = 1000;
		int img_queue_capacity = 1000;		

		cModuleSync* sync;
		cv::Size sensor_dim;
		vector<shared_ptr<CameraLibrary::Camera>> optitrack_cameras;
		vector<shared_ptr<const CameraLibrary::Frame>> optitrack_frames;

		int init_video_mode; //mahdi // {0 = Precision, 1 = Segment, 2 = Object, 3 = MJPEG Mode}
		int init_threshold;
		int init_exposure;
		int init_intensity;
		int init_framerate_mode;    //Mode number: {2 = 100%, 1 = 50%, 0 = 25%} for V120:SLIM cameras
		int init_camera_idx;
		int min_cam_fps = 30; //mahdi for skipping frames
		int init_framerate_value;//mahdi constantly variable framerate
		uint64_t  frame_count; //mahdi frame_count changed from int to uint64 to skip frames for lower framerates below 30
		//mahdi all below
		vector <std::unique_ptr<lpt::concurrent_queue<std::pair<std::string, std::vector<std::string>> >>> camTxtWriteQueues;
		vector <std::unique_ptr<lpt::concurrent_queue<std::pair<std::string, cv::Mat> >>> camImgWriteQueues;
		//concurrent_queue<std::pair<std::string, cv::Mat>> imgWriteQueue; // Queue for images mahdi, one common queue for all cameras
		boost::thread_group camTxtWriterThreads;
		boost::thread_group camImgWriterThreads;
	};

#endif /*USE_NP_CAMERASDK*/

	/*
	 * This class provides the multi threaded streaming pipeline structure of the particle tracking system
	 */
	class StreamingPipeline {
	public:
		typedef std::shared_ptr<lpt::StreamingPipeline> Ptr;
		static StreamingPipeline::Ptr create() { return make_shared< lpt::StreamingPipeline >(); }

		StreamingPipeline() : queue_capacity(200) { shared_objects = std::make_shared < lpt::SharedObjects >(); } //mahdi from 100 to 500
		virtual ~StreamingPipeline() {}

		virtual bool initialize();
		virtual void initializeControlWindow();
		virtual void run();
		virtual void stop();

		virtual void aquireImageData();
		virtual void processImages(int index);
		virtual void solveCorrespondence();
		virtual void reconstuct3DObjects();
		virtual void trackObjects();
		virtual void runControlWindow();
		virtual void runVisualizer() {
			this->visualizer->start();
			std::thread::id this_id = std::this_thread::get_id();
			//std::cout << " mahdi visualizer_thread runVisualizer ID: " << this_id << std::endl;
			cout << "Visualizer thread done" << endl;
			this->visualizer->stop();
		}

		inline void attachCameraSystem(lpt::CameraSystem::Ptr system) {
			this->camera_system.reset();
			this->camera_system = system;
			camera_system->setSharedObjects(shared_objects);
			cout << "Camera System attached to pipeline " << endl;
		}
		inline void attachImageProcessor(lpt::ImageProcessor::Ptr processor) {
			this->processor.reset();
			this->processor = processor;
			cout << "Image Processor attached to pipeline " << endl;
		}

		inline void attachDetector(lpt::Detector::Ptr detector) {
			this->detector.reset();
			this->detector = detector;
			cout << "Object Detector attached to pipeline " << endl;
		}

		inline void attachMatcher(lpt::Correspondence::Ptr matcher) {
			this->matcher.reset();
			this->matcher = matcher;
			matcher->setSharedObjects(shared_objects);
			cout << "Correspondence solver attached to pipeline " << endl;
		}

		inline void attachTracker(lpt::Tracker::Ptr tracker) {
			this->tracker.reset();
			this->tracker = tracker;
			tracker->setSharedObjects(shared_objects);
			cout << "Tracker attached to pipeline " << endl;
		}

		inline void attachVisualizer(lpt::Visualizer::Ptr visualizer) {
			this->visualizer.reset();
			this->visualizer = visualizer;
			visualizer->setSharedObjects(shared_objects);
			cout << "Visualizer attached to pipeline " << endl;
		}

		inline void setSharedObjects(std::shared_ptr<SharedObjects> new_shared_objects) { shared_objects = new_shared_objects; }
		inline std::shared_ptr<SharedObjects> getSharedObjects() { return shared_objects; }

		inline std::shared_ptr<lpt::CameraSystem> getCameraSystem() { return camera_system; }
		inline std::shared_ptr<lpt::Calibrator> getCalibrator() { return calibrator; }

		inline void setInputDataPath(string& path) { shared_objects->input_path = path; }
		inline string getInputDataPath() { return shared_objects->input_path; }

		inline void setOutputDataPath(string& path) { shared_objects->output_path = path; }
		inline string getOutputDataPath() { return shared_objects->output_path; }

		inline void setImageType(cv::Mat image_type) { image_type = image_type; }
		inline cv::Mat getImageType() { return shared_objects->image_type; }

		inline void setFrameRate(int fps) { shared_objects->frame_rate = fps; }
		inline int getFrameRate() { return shared_objects->frame_rate; }

		inline void setKalmanFilter(bool state) { shared_objects->KF_isOn = state; }
		inline bool getKalmanFilter() { return shared_objects->KF_isOn; }

		inline void setQueueCapacity(int capacity) { queue_capacity = capacity; }

		inline void setCamerasStatus(bool state) { cameras_status = state; }
		inline bool getImageViewStatus() const { return image_view_status; }
		inline void setImageViewStatus(bool state) { image_view_status = state; }
		inline bool showCompositeView() const { return composite_view_status; }
		inline void setCompositeView(bool state) { composite_view_status = state; }
		inline bool showDetectionView() const { return detection_view_status; }
		inline void setDetectionView(bool state) { detection_view_status = state; }
		inline bool showReprojectionView() const { return reprojection_view_status; }
		inline void setReprojectionView(bool state) { reprojection_view_status = state; }
		inline bool showTrajectoryView() const { return trajectory_view_status; }
		inline void setTrajectoryView(bool state) { trajectory_view_status = state; }

		friend void callbackRecordVideo(int state, void* data);
		friend void callbackTakeSnapshot(int state, void* data);
		friend void callbackSetImageViewStatus(int state, void* data);
		friend void callbackSetCompositeView(int state, void* data);
		friend void callbackStopCameras(int state, void* data);
		friend void callbackFlushFrameQueue(int state, void* data);
		friend void callbackFlushProcessedQueue(int state, void* data);
		friend void callbackSetDetectionView(int state, void* data);
		friend void callbackSetReprojectionView(int state, void* data);
		friend void callbackSetTrajectoryView(int state, void* data);

		//mahdi 
		inline void setXYRFilter(bool state) { xyr_filter_status = state; }
		inline bool getXYRFilter() const { return xyr_filter_status; }
		friend void callbackSetXYRFilter(int state, void* data);
		friend void callbackSetMinRadius(int state, void* data);//mahdi 
		friend void callbackSetMaxRadius(int state, void* data);//mahdi 
		friend void callbackSet_min_x_pos(int state, void* data);//mahdi 
		friend void callbackSet_max_x_pos(int state, void* data);//mahdi 
		friend void callbackSet_min_y_pos(int state, void* data);//mahdi 
		friend void callbackSet_max_y_pos(int state, void* data);//mahdi 

		void load_Rotation_Matrix();
		void load_Rotation_Matrix_virtual();

		double minParticleRadius = 0.6;
		double maxParticleRadius = 10;
		int minParticleRadius_level = 6;
		int maxParticleRadius_level = 100;
		int minParticle_x_pos = 0;
		int maxParticle_x_pos = 1280;
		int minParticle_y_pos = 0;
		int maxParticle_y_pos = 1024;

	protected:

		std::shared_ptr < lpt::SharedObjects >	shared_objects;
		std::shared_ptr < lpt::CameraSystem >	camera_system;
		std::shared_ptr < lpt::Visualizer >		visualizer;
		std::shared_ptr < lpt::Calibrator >		calibrator;
		std::shared_ptr < lpt::Recorder >		recorder;
		std::shared_ptr < lpt::ImageProcessor >	processor;
		std::shared_ptr < lpt::Detector >		detector;
		std::shared_ptr < lpt::Correspondence >	matcher;
		std::shared_ptr < lpt::Reconstruct3D >	reconstructor;
		std::shared_ptr < lpt::Tracker >			tracker;		

		// Concurrent Queues
		lpt::concurrent_queue < lpt::ImageFrameGroup >	frame_queue;
		lpt::concurrent_queue < lpt::ImageFrameGroup >	processed_queue;
		lpt::concurrent_queue < std::pair<lpt::ImageFrameGroup, vector<lpt::Match::Ptr> > > 	match_queue;
		lpt::concurrent_queue < std::shared_ptr<lpt::Frame3d > >		frame3D_queue;
		lpt::concurrent_queue < std::shared_ptr<lpt::Frame3d > >		monitor_queue;

		boost::thread	imagegrabber_thread;
		boost::thread	matcher_thread;
		boost::thread	reconstructor_thread;
		boost::thread	tracker_thread;
		boost::thread	visualizer_thread;
		boost::thread_group		imageproc_workers_thread_group;

		shared_ptr<boost::barrier> imageproc_barrier;
		boost::mutex imageproc_mutex;

		lpt::ImageFrameGroup imageproc_frames;

		int camera_displayed;       // index of initial camera to be displayed in opencv window
		int view_point;
		int queue_capacity;

		bool cameras_status;
		bool composite_view_status;
		bool image_view_status;
		bool detection_view_status;
		bool reprojection_view_status;
		bool trajectory_view_status;
		bool run_calibration;
		bool xyr_filter_status;
	};

	class Process {
	public:
		Process() {}
		virtual void initialize() {}
		virtual void addControls() {}
		virtual void run() {}
		virtual void shutdown() {}
		virtual void setInputQueue() {}
		virtual void setOuputQueue() {}
		virtual ~Process() {}
	protected:
		boost::thread process_thread;
	};

	class Process1 : public Process {
	public:
		Process1() { }
		virtual void run() { }
		virtual ~Process1() { }
	private:

	};

	class Process2 : public Process {
	public:
		Process2() { }
		virtual void run() { }
		virtual ~Process2() { }
	private:

	};

	class ProcessPipeline {
	public:
		ProcessPipeline() {}
		virtual ~ProcessPipeline() {}

		virtual void addProcess(std::shared_ptr<lpt::Process>& process) {
			vector<std::shared_ptr<lpt::Process>> new_process;
			new_process.push_back(process);
			processes.push_back(new_process);
		}
		virtual void addParallelProcess(vector<std::shared_ptr<lpt::Process> >& process_vector) {
			processes.push_back(process_vector);
		}
	private:
		vector<vector<std::shared_ptr<lpt::Process>>> processes;
	};

} /* NAMESPACE_PT */
#endif /*DATAAQUSITION_H_*/