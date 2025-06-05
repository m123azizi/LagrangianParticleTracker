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
//using namespace std;
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
//	boost::posix_time::microseconds sleeptime(100);
//
//	while (camera_system->areCamerasRunning()) {
//		cout << "inside acquire image data loop " << endl;
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
//			}
//			else {
//				return;
//			}
//		}
//		else
//			boost::this_thread::sleep(sleeptime);
//
//		++frame_count;
//
//	}
//}
//
//void callbackStopCameras(int state, void* data) {
//	lpt::Optitrack* system = static_cast<lpt::Optitrack*> (data);
//	if (system) {
//		system->shutdown();
//	}
//	else
//		cout << "---INVALID pointer to CameraSystem:  Cannot Stop!!" << endl;
//}
//
int main(){
//	std::shared_ptr < lpt::SharedObjects >	shared_objects = make_shared < lpt::SharedObjects >();
//	std::shared_ptr < lpt::Optitrack >	camera_system = lpt::Optitrack::create();
//	camera_system->setSharedObjects(shared_objects);
//
//	bool cameras_ok = false;
//
//	if (camera_system)
//		cameras_ok = camera_system->initializeCameras();
//
//	if (!cameras_ok) {
//		cout << "Cameras NOT initialized, exiting." << endl;
//		return 0;
//	}
//
//	this_thread::sleep_for(chrono::milliseconds(3000));
//	
//	// initialize control window 	
//	auto& cameras = shared_objects->cameras;
//	vector<shared_ptr<CameraLibrary::Camera>> optitrack_cameras = camera_system->getOptitrackCameras();
//
//	for (int i = 0; i < optitrack_cameras.size(); i++) {
//		int camera_id = i;
//		optitrack_cameras[camera_id]->SetVideoType(Core::MJPEGMode);
//		optitrack_cameras[camera_id]->SetExposure(4000);
//	}
//	
//	if (cameras.empty()) {
//		cout << "Could not initialize control window: No cameras found" << endl;
//		return 0;
//	}
//
//	cout << "Initializing Control Window with " << cameras.size() << " Cameras" << endl;
//	// Set up display window using opencv
//	string null;
//	void* camerasys_void_ptr = static_cast<void*> (&*camera_system);
//
//	int camera_displayed = 0;       // index of initial camera to be displayed in opencv window
//	cv::namedWindow(camera_system->getWindowName());
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
//	this_thread::sleep_for(chrono::milliseconds(3000));
//
//	boost::thread imagegrabber_thread = boost::thread(aquireImageData, shared_objects, camera_system, boost::ref(frame_queue), boost::ref(frame_count));
//	cout << "-----------thread for acquire ImageData is created!" << endl;
//	
//	this_thread::sleep_for(chrono::milliseconds(300));
//
//	cout << "queue size is: " << frame_queue.size() << endl;
//
//	// run control window
//	while (camera_system->areCamerasRunning() && !frame_queue.empty()) {
//		lpt::ImageFrameGroup frame_group;
//		if (frame_queue.try_pop(frame_group)) {
//			cout << "Succeed to pop out frame group" << endl;
//			cv::imshow(window_name, frame_group[camera_displayed].image);
//			cv::waitKey(33);
//		}
//		else {
//			cout << "Failed to pop out frame group" << endl;
//		}
//
//	}
//
//	imagegrabber_thread.join();
//	camera_system->shutdown();
//
	return 0;
}