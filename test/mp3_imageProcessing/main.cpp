//#include <iostream>
//#include "cameralibrary.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <core.hpp>
//#include <dataaquisition.hpp>
//#include <chrono>
//#include <boost/thread.hpp>
//#include <boost/thread/barrier.hpp>
//#include <boost/chrono.hpp>
//#include <boost/atomic.hpp>
//
//#define NUM_CAMERAS 8
//
//using namespace std;
//
//bool checkSynchronization(std::shared_ptr < lpt::SharedObjects > shared_objects, 
//	std::shared_ptr < lpt::Optitrack > camera_system, int tempTimes=5) 
//{
//	//bool success = false;
//	boost::posix_time::milliseconds sleeptime(2000);
//	auto sync = camera_system->getSyncModule();
//	auto& cameras = shared_objects->cameras;
//
//	for (int i = 0; i < tempTimes; i++) {
//		CameraLibrary::FrameGroup* native_frame_group = sync->GetFrameGroup();
//		if (native_frame_group && native_frame_group->Count() == cameras.size()) {
//			if (sync->LastFrameGroupMode() != CameraLibrary::FrameGroup::Hardware) {
//				cout << "\t Cameras LastFrameGroupMode NOT equal to \"HardWare\" at " << i << "'th temp." << endl;
//			}
//			else {
//				cout << "\t Camera SUCCESSFULLY Synchronized." << endl;
//				return true;
//			}
//		}
//		else {
//			cout << "\t native frame group count NOT equal to cameras' size" << endl;
//		}
//
//		boost::this_thread::sleep(sleeptime);
//
//	}
//
//	return false;
//}
//
//void aquireImageData(std::shared_ptr < lpt::SharedObjects > shared_objects,
//	std::shared_ptr < lpt::Optitrack > camera_system,
//	lpt::concurrent_queue < lpt::ImageFrameGroup >& frame_queue,
//	int& frame_count)
//{
//	cout << "inside acquire image data" << endl;
//
//	auto sync = camera_system->getSyncModule();
//	auto& cameras = shared_objects->cameras;
//	boost::posix_time::microseconds sleeptime(10);
//
//	while (camera_system->areCamerasRunning()) {
//		lpt::ImageFrameGroup frame_group(cameras.size());
//
//		// grab frame group
//		shared_ptr<const CameraLibrary::FrameGroup> native_frame_group = sync->GetFrameGroup();
//
//		if (native_frame_group && native_frame_group->Count() == shared_objects->cameras.size()) {
//			auto& image_type = shared_objects->image_type;
//			auto& half_image_type = shared_objects->half_image_type;
//
//			auto& optitrack_frames = camera_system->getOptitrackFrames();
//
//			for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id) {
//				optitrack_frames[camera_id] = native_frame_group->GetFrame(camera_id);
//				frame_group[camera_id].frame_index = optitrack_frames[camera_id]->FrameID();
//				int frame_id = frame_group[camera_id].frame_index;
//
//				cv::Mat temp = image_type.clone();
//				optitrack_frames[camera_id]->Rasterize(image_type.size().width, image_type.size().height,
//					static_cast<unsigned int>(temp.step), 8, temp.data);
//
//				cv::Rect ROI(0, 0, image_type.size().width / 2, image_type.size().height / 2);
//				cv::resize(temp(ROI), frame_group[camera_id].image, image_type.size(), 0.0, 0.0, cv::INTER_CUBIC);
//				optitrack_frames[camera_id]->Release();
//
//			}
//			native_frame_group->Release();
//			if (sync->LastFrameGroupMode() != CameraLibrary::FrameGroup::Hardware) {
//				for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id)
//					frame_group[camera_id].image = image_type.clone();
//				cout << "\t Cameras NOT Synchronized: Frame # = " << frame_count << endl;
//			}
//
//			if (!frame_queue.full()) {
//				frame_queue.push(std::move(frame_group));
//				cout << "frame queue count is: " << frame_count << " queue size is: " << frame_queue.size() << endl;
//				return;
//			}
//			++frame_count;
//
//		}
//		else
//			boost::this_thread::sleep(sleeptime);
//
//
//	}
//}
//
//void callbackStopCameras(int state, void* data) {
//	lpt::Optitrack* system = static_cast<lpt::Optitrack*> (data);
//	if (system) {
//		system->setCamerasStatus(false);
//		system->shutdown();
//	}
//	else 
//		cout << "---INVALID pointer to CameraSystem:  Cannot Stop!!" << endl;
//}
//
//void shutdownCameraSystem(std::shared_ptr < lpt::Optitrack > camera_system) {
//	if (camera_system) {
//		camera_system->setCamerasStatus(false);
//		camera_system->shutdown();
//	}
//	else
//		cout << "---INVALID pointer to CameraSystem:  Cannot Stop!!" << endl;
//}
//
//bool initiateCameraSystem(std::shared_ptr < lpt::SharedObjects > shared_objects,
//	std::shared_ptr < lpt::Optitrack > camera_system, int maxTemp=5) 
//{
//	this_thread::sleep_for(chrono::milliseconds(5000));
//
//	bool cameras_ok = false;
//	if (camera_system)
//		cameras_ok = camera_system->initializeCameras();
//
//	if (!cameras_ok) {
//		cout << "Cameras NOT initialized, exiting." << endl;
//		return false;
//	}
//
//
//	// initialize control window 	
//	auto& cameras = shared_objects->cameras;
//
//	if (cameras.empty()) {
//		cout << "Could not initialize control window: No cameras found" << endl;
//		return false;
//	}
//
//	if (cameras.size() != NUM_CAMERAS) {
//		cout << "Number of initialized Cameras not equal to pre-defined number" << endl;
//		return false;
//	}
//
//	return true;
//}
//
//void process_image(cv::Mat& image) {
//	cv::threshold(image, image, 50, 255, 0);
//}
//
int main(){
//	std::shared_ptr < lpt::SharedObjects >	shared_objects = make_shared < lpt::SharedObjects >();
//	std::shared_ptr < lpt::Optitrack >	camera_system = lpt::Optitrack::create();
//	camera_system->setSharedObjects(shared_objects);
//
//	bool success = initiateCameraSystem(shared_objects, camera_system);
//
//	bool synchronized = false;
//
//	if (success) {
//		synchronized = checkSynchronization(shared_objects, camera_system, 10);	
//	}
//	else {
//		cout << "Error initialing camera system, exiting..." << endl;
//		return 0;
//	}
//
//	if (!synchronized) {
//		cout << "Cannot synchronize cameras, exiting...." << endl;
//		return 0;
//	}
//
//	this_thread::sleep_for(chrono::milliseconds(3000));
//
//	vector<shared_ptr<CameraLibrary::Camera>> optitrack_cameras = camera_system->getOptitrackCameras();
//
//	try {
//		for (int i = 0; i < optitrack_cameras.size(); i++) {
//			int camera_id = i;
//			//optitrack_cameras[camera_id]->SetVideoType(Core::MJPEGMode);
//			optitrack_cameras[camera_id]->SetExposure(4000);
//		}
//	}
//	catch (const exception& e) {
//		std::cout << "Error exists when adjusting camera parameters " << e.what() << endl;
//	}
//
//	auto& cameras = shared_objects->cameras;
//	cout << "Initializing Control Window with " << cameras.size() << " Cameras" << endl;
//	// Set up display window using opencv
//	string null;
//	void* camerasys_void_ptr = static_cast<void*> (&*camera_system);
//
//	int camera_displayed = 0;       // index of initial camera to be displayed in opencv window
//	cv::namedWindow(camera_system->getWindowName(), cv::WINDOW_NORMAL);
//	cv::createTrackbar("Camera", camera_system->getWindowName(), &camera_displayed, static_cast<int>(cameras.size() - 1), 0);
//	cv::createButton("Stop Cameras", callbackStopCameras, camerasys_void_ptr, cv::QT_PUSH_BUTTON, 0);
//
//	int frame_count = 0;
//
//	int queue_capacity = 1000;
//	string window_name = camera_system->getWindowName();
//	lpt::concurrent_queue < lpt::ImageFrameGroup >	frame_queue;
//	frame_queue.setCapacity(queue_capacity);
//
//	boost::thread imagegrabber_thread = boost::thread(aquireImageData, shared_objects, camera_system, boost::ref(frame_queue), boost::ref(frame_count));
//	cout << "-----------thread for acquire ImageData is created!" << endl;
//	
//	this_thread::sleep_for(chrono::milliseconds(8000));
//
//	cout << "queue size is: " << frame_queue.size() << endl;
//
//	// run control window
//	while (camera_system->areCamerasRunning()) {
//		lpt::ImageFrameGroup frame_group;
//		if (frame_queue.try_pop(frame_group)) {
//			cout << "Succeed to pop out frame group" << endl;
//
//			// perform image processing
//			for (int i = 0; i < frame_group.size(); i++) {
//				auto& image = frame_group[i].image;
//				process_image(image);
//			}
//
//			cv::imshow(window_name, frame_group[camera_displayed].image);
//			cv::waitKey(33);
//		}
//		else {
//			cout << "Failed to pop out frame group, CLOSING camera system" << endl;
//
//			camera_system->setCamerasStatus(false);
//		}
//
//	}
//
//	//imagegrabber_thread.join();
//	camera_system->shutdown();
//
	return 0;
}