/*
Real-time particle tracking system:
This is the main code for the real-time particle tracking system.
This creates the streaming pipeline of tasks including:
	1) data aquisition from camera
	2) image processing
	3) object detection in each image
	4) camera correspondence solver
	5) 3D reconstruction
	6) Temporal tracking
	7) Visualization

 copyright: Douglas Barker 2011
*/

#include "core.hpp"
#include "imageproc.hpp"
#include "datagen.hpp"
#include "dataaquisition.hpp"
#include "correspond.hpp"
#include "tracking.hpp"
#include "visualization.hpp"
#include <iostream>

//chages by mahdi
#include <iomanip> //I think for text file precesion writing


using namespace std;

int main(int argc, char** argv) {
	cout << fixed; //mahdi
	cout << " the code have argc= " << argc << " arguments /n";
	for (int i = 0; i < argc; ++i) {
		cout << " argv at i= " << i << " is " << argv[i] << endl;
	}	
	int n = boost::thread::hardware_concurrency();
	cout << "number of threads available: " << n << endl;
	//freopen("outMahdi.txt", "w", stdout); // mahdi 
	string input = (argc > 1 ? argv[1] : "../../../data/input/");
	string output = (argc > 2 ? argv[2] : "../../../data/output/");

	//string cameras_file = input + "n12c_after_3_point_line.yaml";
	string cameras_file = input + "n11c.yaml";
	//string camera_pairs_file = input + "n14pfpga.yaml";
	string camera_pairs_file = input + "n11p.yaml";
	string grid_data_file = input + "GRID_DATA.yaml";

	lpt::StreamingPipeline pipeline;
	pipeline.setQueueCapacity(1000);

	auto processor = lpt::ImageProcessor::create();
	lpt::ImageProcess::Ptr blur = lpt::GaussianBlur::create(3);    //# TODO uncomment gaussian blur later if not using svm
	lpt::ImageProcess::Ptr thresh = lpt::Threshold::create(50);  // uncomment when using FC detector
	processor->addProcess(blur);
	processor->addProcess(thresh);

	auto detector = lpt::FindContoursDetector::create();

	auto camera_system = lpt::Optitrack::create();

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

	auto tracker = lpt::Tracker::create();
	tracker->setCostCalculator(lpt::CostMinimumAcceleration::create());
	tracker->params.min_radius = 4.0; //mm
	tracker->params.min_radius_level = 4;
	tracker->params.max_radius = 25.0; //mm
	tracker->params.max_radius_level = 25;
	tracker->params.KF_sigma_a = 2.75E-4;
	tracker->params.KF_sigma_z = 1E-2;
	//
	bool KalmanFilter = false; // mahdi kalman filter is here

	auto visualizer = lpt::Visualizer::create();
	visualizer->readGridData(grid_data_file);
	visualizer->params.queue_capacity = 1000;//by mahdi

	pipeline.setInputDataPath(input);
	pipeline.setOutputDataPath(output);
	pipeline.load_Rotation_Matrix();
	pipeline.setKalmanFilter(KalmanFilter);
	pipeline.attachCameraSystem(camera_system);
	pipeline.attachImageProcessor(processor);
	pipeline.attachDetector(detector);
	pipeline.attachMatcher(matcher_cuda);
	pipeline.attachTracker(tracker);
	pipeline.attachVisualizer(visualizer);

	bool on = pipeline.initialize();

	if (on) {
		camera_system->loadCameraParams(cameras_file);
		camera_system->loadCameraPairParams(camera_pairs_file);
		pipeline.run();
	}
	else {
		cout << "System could not initialize: Shutting down" << endl;
		system("pause");
	}
	cout.clear();
	cout << "Finished" << endl;
	return 0;
}

