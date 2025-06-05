#include <core.hpp>
#include <imageproc.hpp>
#include <datagen.hpp>
#include <dataaquisition.hpp>
#include <tracking.hpp>

//#include "core.hpp"
//#include "imageproc.hpp"
//#include "datagen.hpp"
//#include "dataaquisition.hpp"
//#include "correspond.hpp"
//#include "tracking.hpp"
//#include "visualization.hpp"
//#include <iostream>

using namespace std;

int main(int argc, char** argv) {
	cout << fixed; //mahdi
	cout << "runnig virtual camera from saved files" << endl;

	//freopen("outMahdi.txt", "w", stdout); // mahdi 
	string input = (argc > 1 ? argv[1] : "../../../data/input/");
	string output = (argc > 2 ? argv[2] : "../../../data/output/");

	string cameras_file = input + "n11c.yaml";
	string camera_pairs_file = input + "n11p.yaml";
	string grid_data_file = input + "GRID_DATA.yaml";

	string image_dir = output + "calib_images";
	//string text_dir = "C:\\mahdi\\VPTV_VS2022\\buildbroken\\bin\\Release\\txto_30min test of rotational with writing 30fps";
	string text_dir = "C:\\mahdi\\my_experiments\\myFirstJetTest\\V4_final_V4_4rounds\\txto_4 rounds\\separated\\outputs"; // my first jet test
	//string text_dir = "C:\\mahdi\\VPTV_VS2022\\data\\output\\txto\\from breanna with missing to test\\verified resuilt filtered_frames";
	//string text_dir = "C:\\mahdi\\my_experiments\\my2ndJet\\data\\filtered_frames"; //240 fps data

	//int frame_start = 172442; //240 fps data
	//int frame_end = 603011; //240 fps data

	//int frame_start = 866; //breanna's test
	//int frame_end = 893; //breanna's test
	
	int frame_start = 259313; // my first jet test
	int frame_end = 506910; // my first jet test
	int frame_step = 1;
	int vfps = 120;


	lpt::StreamingPipeline pipeline;
	pipeline.setQueueCapacity(1000);

	lpt::ImageProcessor::Ptr processor = lpt::ImageProcessor::create();
	lpt::ImageProcess::Ptr blur = lpt::GaussianBlur::create(3);
	lpt::ImageProcess::Ptr thresh = lpt::Threshold::create(50);
	processor->addProcess(blur);
	processor->addProcess(thresh);

	lpt::FindContoursDetector::Ptr detector = lpt::FindContoursDetector::create();
	//lpt::GoodFeaturesToTrackDetector::Ptr detector = lpt::GoodFeaturesToTrackDetector::create();

	//mahdi: main code makes a camera_system here but this one doesn't
	//difference with main code is as follows:

	//int image_width = 1280;
	//int image_height = 1024;

	//auto image_creator = std::make_shared<lpt::ImageCreator>();
	//image_creator->image_type = cv::Mat::zeros(cv::Size(image_width, image_height), CV_8UC1);

	//image_creator->radius = 0;//0;
	//image_creator->intensity = 0;
	//image_creator->object_intensity = 5E8;
	//image_creator->object_size = 3;
	//image_creator->blur_ksize = 3;

	//then it makes a virtual camera system
	//auto camera_system = lpt::SINTC_VirtualCamerasFile::create(cameras_file, image_dir, text_dir, frame_start, frame_end, frame_step, vfps);
	auto camera_system = lpt::NICT_VirtualCamerasFile::create(cameras_file, image_dir, text_dir, frame_start, frame_end, frame_step, vfps);

	//auto matcher = lpt::PointMatcher::create();
	//matcher->params.match_threshold = 5.0;
	//matcher->params.match_thresh_level = 50;
	//matcher->params.threshold_distance = 10;
	//matcher->params.threshold_distance_level = 100;

	auto matcher_cuda = lpt::PointMatcherCUDA::create();
	matcher_cuda->params.match_threshold = 5.0; //pixels
	matcher_cuda->params.match_thresh_level = 50;
	matcher_cuda->params.threshold_distance = 10;
	matcher_cuda->params.threshold_distance_level = 100;

	lpt::Tracker::Ptr tracker = lpt::Tracker::create();
	//tracker->setCostCalculator(lpt::CostMinimumAcceleration::create());
	tracker->setCostCalculator(lpt::DirectionalCostMinAcc::create());
	tracker->params.min_radius = 4.0; //mm
	tracker->params.min_radius_level = 4;
	tracker->params.max_radius = 25.0; //mm
	tracker->params.max_radius_level = 25;
	tracker->params.KF_sigma_a = 1E-5;
	tracker->params.KF_sigma_z = 1E-1;

	bool KalmanFilter = false;

	lpt::Visualizer::Ptr visualizer = lpt::Visualizer::create();

	visualizer->readGridData(grid_data_file);
	visualizer->params.queue_capacity = 1000;

	pipeline.setInputDataPath(input);
	pipeline.setOutputDataPath(output);
	pipeline.setKalmanFilter(KalmanFilter);
	pipeline.attachCameraSystem(camera_system);
	pipeline.attachImageProcessor(processor);
	pipeline.attachDetector(detector);

	pipeline.load_Rotation_Matrix();//diff here, main code loads rotation matrix but this one loads virtual. mahdi changed it to the original form

	//pipeline.attachMatcher(matcher);

	pipeline.attachMatcher(matcher_cuda);

	pipeline.attachTracker(tracker);
	pipeline.attachVisualizer(visualizer);

	bool on = pipeline.initialize();

	if (on) {
		camera_system->loadCameraParams(cameras_file);
		camera_system->loadCameraPairParams(camera_pairs_file);
		pipeline.run();
	}
	else
		cout << "System could not initialize: Shutting down" << endl;

	cout << "Finished Stream Data" << endl;
	return 0;
}