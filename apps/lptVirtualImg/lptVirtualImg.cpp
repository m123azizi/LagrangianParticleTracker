#include <core.hpp>
#include <imageproc.hpp>
#include <datagen.hpp>
#include <dataaquisition.hpp>
#include <tracking.hpp>

using namespace std;

int main(int argc, char** argv) {
	cout << fixed; //mahdi
	cout << "virtual camera from saved IMG files" << endl;

	//freopen("outMahdi.txt", "w", stdout); // mahdi 
	string input = (argc > 1 ? argv[1] : "../../../data/input/");
	string output = (argc > 2 ? argv[2] : "../../../data/output/");

	string cameras_file = input + "n8c.yaml";
	string camera_pairs_file = input + "n8p.yaml";
	string grid_data_file = input + "GRID_DATA.yaml";

	//string image_dir = output + "ImgFileReader";
	string image_dir = "C:\\mahdi\\my_experiments\\breanna_labeled images\\URSA\\fall24\\30";
	//int frame_start = 115890;
	//int frame_end = 120338; //the last index
	int frame_start = 13035;
	int frame_end = 13063;
	int frame_step = 1;
	int vfps = 120;


	lpt::StreamingPipeline pipeline;
	pipeline.setQueueCapacity(1000);

	lpt::ImageProcessor::Ptr processor = lpt::ImageProcessor::create();
	lpt::ImageProcess::Ptr blur = lpt::GaussianBlur::create(3);
	lpt::ImageProcess::Ptr thresh = lpt::Threshold::create(20);
	processor->addProcess(blur);
	processor->addProcess(thresh);

	lpt::FindContoursDetector::Ptr detector = lpt::FindContoursDetector::create();

	//both lines below equal
	//auto camera_system = std::make_shared<lpt::VirtualCamerasImg>(cameras_file, image_dir, frame_start, frame_end, frame_step, vfps);
	lpt::VirtualCamerasImg::Ptr camera_system = std::make_shared<lpt::VirtualCamerasImg>(cameras_file, image_dir, frame_start, frame_end, frame_step, vfps);

	auto matcher = lpt::PointMatcher::create();
	matcher->params.match_threshold = 5.0;
	matcher->params.match_thresh_level = 50;
	matcher->params.threshold_distance = 10;
	matcher->params.threshold_distance_level = 100;

	auto matcher_cuda = lpt::PointMatcherCUDA::create();
	matcher_cuda->params.match_threshold = 5.0; //pixels
	matcher_cuda->params.match_thresh_level = 50;
	matcher_cuda->params.threshold_distance = 10;
	matcher_cuda->params.threshold_distance_level = 100;

	lpt::Tracker::Ptr tracker = lpt::Tracker::create();
	tracker->setCostCalculator(lpt::CostMinimumAcceleration::create());
	//tracker->setCostCalculator(lpt::CostNearestNeighbor::create());
	tracker->params.min_radius = 4.0; //mm
	tracker->params.min_radius_level = 4;
	tracker->params.max_radius = 25.0; //mm
	tracker->params.max_radius_level = 25;
	tracker->params.KF_sigma_a = 1E-5;
	tracker->params.KF_sigma_z = 1E-1;

	bool KalmanFilter = false;

	lpt::Visualizer::Ptr visualizer = lpt::Visualizer::create();
	//visualizer->getVolumeGrid()->setGridOrigin(-600,0.0,-250);
	//visualizer->getVolumeGrid()->setGridDimensions(1500, 1500, 500); // mm
	//visualizer->getVolumeGrid()->setGridCellCounts(210, 210, 1); //fine grid
	//visualizer->getVolumeGrid()->setGridCellCounts(70, 70, 1); //coarse grid

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