/*
Data aquisition module implementation:
These classes provide data aquisition funcionality to the particle tracking system and the overall multi-threaded streaming framework.

copyright: Douglas Barker 2011
*/

#include "dataaquisition.hpp"

namespace lpt {

	using namespace std;
	using namespace CameraLibrary;

	/*****lpt::Video class implementation*****/

	void Video::writeAVI(int playback_fps, int codec) {
		cv::Size frame_size = video_frames[0].size();
		bool color = (video_frames[0].channels() == 3) ? true : false;
		cv::VideoWriter writer(file_path, codec, playback_fps, frame_size, color);
		for (int f = 0; f < video_frames.size(); ++f)
			writer.write(video_frames[f]);
		cout << "Wrote video to file: " << file_path << endl;
	}

	void Video::writeImages(string basename, string filetype) {
		for (int f = 0; f < video_frames.size(); ++f) {
			stringstream file_name;
			file_name << file_path << basename << "_" << f << filetype;
			cv::imwrite(file_name.str(), video_frames[f]);
		}
	}

	void Video::readAVI() {
		if (!file_path.empty()) {
			cout << "reading " << file_path << endl;
			cv::VideoCapture capture(file_path);
			cv::Mat frame;
			while (capture.read(frame)) {
				cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
				this->addImageToVideo(frame.clone());
			}
		}
	}

	void Video::readImages(string basename, int numframes, string filetype) {
		video_frames.resize(numframes);
		for (int f = 0; f < video_frames.size(); ++f) {
			stringstream file_name;
			file_name << file_path << basename << "_" << f << filetype;
			video_frames[f] = cv::imread(file_name.str());
		}
	}


	/*****lpt::Recorder class implementation*****/

	void Recorder::takeSnapShot(vector<cv::Mat>& frames) {
		snapshot_id++;
		for (int camera_id = 0; camera_id < frames.size(); camera_id++) {
			stringstream filename;
			filename << path << camera_id << "_" << snapshot_id << ".jpg";
			cv::imwrite(filename.str(), frames[camera_id]);
			imagelists[camera_id].push_back(std::move(filename.str()));
		}
		snapshot_requested = false;
		cout << "Snapshot taken: ID = " << snapshot_id << endl;
	}

	void Recorder::writeSnapShotImageLists() {
		stringstream mainlistname;
		mainlistname << path << "mainlist.txt";
		ofstream mainlistfile(mainlistname.str().c_str());
		for (int camera_id = 0; camera_id < imagelists.size(); camera_id++) {
			stringstream listname;
			listname << path << "imagelist" << camera_id << ".txt";
			mainlistfile << listname.str() << endl;
			ofstream imagelistfile(listname.str().c_str());
			for (int image_id = 0; image_id < imagelists[camera_id].size(); image_id++) {
				imagelistfile << imagelists[camera_id][image_id] << endl;
			}
			imagelistfile.close();
		}
		mainlistfile.close();
	}

	void Recorder::createVideos() {
		video_id++;
		for (int cam_id = 0; cam_id < number_of_cameras; cam_id++) {
			stringstream filename;
			filename << path << cam_id << "_" << video_id << ".avi";
			Video new_video(filename.str());
			videos[cam_id].push_back(std::move(new_video));
		}
		record_video = true;
		cout << "Recording video clip (" << clip_length << " frames): ID = " << video_id << endl;
	}

	void Recorder::addFramesToVideos(vector<cv::Mat>& frames) {
		if (videos[0][video_id].getVideoLength() < clip_length) {
			for (int cam_id = 0; cam_id < frames.size(); ++cam_id)
				videos[cam_id][video_id].addImageToVideo(frames[cam_id]);
		}
		else {
			record_video = false;
			cout << "Finished recording video clip: ID = " << video_id << endl;
		}
	}

	void Recorder::writeVideosToFile(int fps, int codec) {
		for (int cam_id = 0; cam_id < number_of_cameras; cam_id++)
			for (int count = 0; count < videos[cam_id].size(); count++)
				videos[cam_id][count].writeAVI(fps, codec);
	}

	void callbackRecordVideo(int state, void* data) {
		Recorder* recorder = static_cast<Recorder*> (data);
		if (!recorder->isVideoRecording())
			recorder->createVideos();
	}

	void callbackTakeSnapshot(int state, void* data) {
		Recorder* recorder = static_cast<Recorder*> (data);
		recorder->requestSnapShot();
	}

	/*****lpt::CameraSystem class implementation*****/

	void CameraSystem::loadCameraParams(const string filepath) {
		auto& cameras = shared_objects->cameras;
		if (cameras.empty()) {
			cout << "pt_cameras is empty: cannot load camera data" << endl;
			return;
		}
		vector<lpt::Camera> cameras_data;
		lpt::readCamerasFile(filepath, cameras_data);

		if (cameras_data.size() == cameras.size()) {
			for (int a = 0; a < cameras.size(); ++a) {
				for (int b = 0; b < cameras_data.size(); ++b) {
					if (cameras[a].name == cameras_data[b].name) {
						//The cameras match - Load camera object with matched data from file
						auto frames = cameras[a].frames;
						cameras[a] = cameras_data[b];
						cameras[a].frames = frames;
						break;
					}
				}
			}
			cout << "Loaded camera parameters for " << cameras_data.size() << " cameras: " << endl;
		}
	}

	void CameraSystem::loadCameraPairParams(const string filepath) {
		auto& cameras = shared_objects->cameras;
		auto& camera_pairs = shared_objects->camera_pairs;
		if (cameras.empty()) {
			cout << "cameras is empty: cannot load camera pair data" << endl;
			return;
		}

		lpt::readCameraPairsFile(filepath, cameras, camera_pairs);  //TODO: This will fail if the file contains pairs for a greater number of cameras than pt_cameras has
		cout << "Loaded camera parameters for " << camera_pairs.size() << " camera pairs: " << endl;
	}


	/*****lpt::VirtualCameras class implementation*****/

	bool VirtualCameras::initializeCameras() {
		cout << "Virtual Camera System initializing" << endl;

		auto& cameras = shared_objects->cameras;
		auto& camera_pairs = shared_objects->camera_pairs;

		shared_objects->camera_type = lpt::VIRTUAL;
		lpt::readCamerasFile(cameras_file, cameras);

		this->generator->setSharedObjects(this->shared_objects);

		generator->setDataPath(shared_objects->output_path);
		generator->read3DTrajectoryFile(trajectory_file, lpt::PLAINTEXT, 0);
		generator->project3DFramesTo2D();
		shared_objects->frame_rate = 30;
		shared_objects->image_type = this->generator->getImageCreator()->image_type;
		cout << shared_objects->image_type.size().height << endl << endl;

		//// push back the get particle collection from camera
		//bool getParticleFromCamera = false;
		//for (int i = 0; i < cameras.size(); ++i) {
		//	collect_particledata_from_camera_vec.push_back(getParticleFromCamera);
		//	cout << "The " << i << " th" << " camera" << " get particle from camera? " << collect_particledata_from_camera_vec[i] << " " << endl;
		//}

		if (!cameras[0].frames.empty()) {
			this->setCamerasStatus(true);
			return true;
		}
		else
			return false;
	}

	void VirtualCameras::addControls() {
		string null = "";
		cv::createTrackbar("FrameRate", null, &frame_rate_level, 50000, 0, 0);
		cv::createTrackbar("Light intensity", null, &this->generator->getImageCreator()->object_intensity, 10E10, 0, 0);
		//	cv::createTrackbar("Object size (mm)", null , &this->generator->getImageCreator()->object_size, 20, 0, 0);   //TODO: allow object size to be adjusted during simulation
		cout << "Virtual Camera Controls added" << endl;
		collect_particledata_from_camera = false;
	}

	bool VirtualCameras::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;

		for (int c = 0; c < cameras.size(); ++c) {
			lpt::ImageFrame& frame = cameras[c].frames[frame_index];
			frame_group[c].frame_index = frame.frame_index;
			frame_group[c].image = frame.image.clone();
		}

		if (frame_index < cameras[0].frames.size() - 1)
			++frame_index;
		else
			frame_index = 0;

		boost::posix_time::microseconds sleeptime(this->frame_rate_level);
		boost::this_thread::sleep(sleeptime);

		return true;
	}

	void VirtualCameras::shutdown() {
		cout << "\t-------VirtualCameras Shutdown Complete" << endl;
	}


	/***** mahdi : lpt::VirtualCamerasFile class implementation*****/
	bool SIST_VirtualCamerasFile::initializeCameras() {
		cout << "SIST Virtual File Camera System initializing" << endl;
		auto& cameras = shared_objects->cameras;
		//auto& camera_pairs = shared_objects->camera_pairs;
		shared_objects->camera_type = lpt::VIRTUAL;
		shared_objects->image_type = cv::Mat::zeros(cv::Size(1280, 1024), CV_8UC1);
		lpt::readCamerasFile(camerasfile, cameras);
		lpt::ImageFrame init_frame; //setting the beging framegroup an empty one to start the cameras
		init_frame.image = shared_objects->image_type; init_frame.frame_index = 123;
		for (int c = 0; c < cameras.size(); c++)
			cameras[c].frames.push_back(init_frame);
		//std::system("pause");

		shared_objects->frame_rate = virtual_fps;//maybe need to modify later
		cout << " virtual_fps = " << virtual_fps << endl;

		if (!cameras[0].frames.empty()) {
			cout << " *************camera status set to true************** " << endl;
			this->setCamerasStatus(true);
			//std::system("pause");
			return true;
		}
		else
			return false;
	}

	bool NICT_VirtualCamerasFile::initializeCameras() {
		cout << "NICT Virtual File Camera System initializing" << endl;
		auto& cameras = shared_objects->cameras;
		//auto& camera_pairs = shared_objects->camera_pairs;
		shared_objects->camera_type = lpt::VIRTUAL;
		shared_objects->image_type = cv::Mat::zeros(cv::Size(1280, 1024), CV_8UC1);
		lpt::readCamerasFile(camerasfile, cameras);
		lpt::ImageFrame init_frame; //setting the beging framegroup an empty one to start the cameras
		init_frame.image = shared_objects->image_type; init_frame.frame_index = 123;
		for (int c = 0; c < cameras.size(); c++)
			cameras[c].frames.push_back(init_frame);
		//std::system("pause");
		input_txt_files.resize(cameras.size());
		input_txt_files_last_positions.resize(cameras.size(), std::streampos(0));
		// Open all camera text files
		for (int camera_id = 0; camera_id < cameras.size(); ++camera_id) {
			std::string textFilePath = text_dir + "/cam" + std::to_string(camera_id) + ".txt";
			input_txt_files[camera_id].open(textFilePath);
			cout << " file open textFilePath= " << textFilePath << endl;
			if (!input_txt_files[camera_id].is_open()) {
				std::cout << "Failed to open text file: " << textFilePath << std::endl;
				return false;
			}
		}
		shared_objects->frame_rate = virtual_fps;//maybe need to modify later
		cout << " virtual_fps = " << virtual_fps << endl;

		if (!cameras[0].frames.empty()) {
			cout << " *************camera status set to true************** " << endl;
			this->setCamerasStatus(true);
			//std::system("pause");
			return true;
		}
		else
			return false;
	}

	bool SINTC_VirtualCamerasFile::initializeCameras() {
		cout << "SINTC Virtual File Camera System initializing" << endl;
		auto& cameras = shared_objects->cameras;
		//auto& camera_pairs = shared_objects->camera_pairs;
		shared_objects->camera_type = lpt::VIRTUAL;
		shared_objects->half_image_type = cv::Mat::zeros(cv::Size(640, 512), CV_8UC1);
		shared_objects->image_type = cv::Mat::zeros(cv::Size(1280, 1024), CV_8UC1);
		lpt::readCamerasFile(camerasfile, cameras);
		lpt::ImageFrame init_frame; //setting the beging framegroup an empty one to start the cameras
		init_frame.image = shared_objects->half_image_type; init_frame.frame_index = 123;
		for (int c = 0; c < cameras.size(); c++)
			cameras[c].frames.push_back(init_frame);
		//std::system("pause");

		shared_objects->frame_rate = virtual_fps;//maybe need to modify later
		cout << " virtual_fps = " << virtual_fps << endl;
		setParticleCollectionFromCamera(false);
		if (!cameras[0].frames.empty()) {
			cout << " *************camera status set to true************** " << endl;
			this->setCamerasStatus(true);
			// List all files and sort them in the constructor
			for (int cam_id = 0; cam_id < shared_objects->cameras.size(); ++cam_id) {
				string imagePattern = "calib_camera" + to_string(cam_id) + "_";
				for (boost::filesystem::directory_iterator itr(image_dir); itr != boost::filesystem::directory_iterator(); ++itr) {
					string fileName = itr->path().filename().string();
					if (fileName.find(imagePattern) == 0 && itr->path().extension() == ".png") {
						file_map[cam_id].push_back(itr->path().string());
					}
				}
				std::sort(file_map[cam_id].begin(), file_map[cam_id].end());

				file_index_map[cam_id] = 0; // Initialize file index for each camera
			}
			// Print all file names in the file map
			for (const auto& entry : file_map) {
				int cam_id = entry.first;
				const auto& files = entry.second;
				cout << "Camera ID " << cam_id << " has " << files.size() << " files:\n";
				for (const auto& file : files) {
					//cout << file << endl;
				}
			}
			//std::system("pause");
			return true;
		}
		else
			return false;
	}

	void VirtualCamerasFile::addControls() {
		string null = "";
		//const std::string windowName = this->getWindowName();
		//camera_system->getWindowName()
		cv::createTrackbar("FrameRate", this->getWindowName(), &frame_rate_level, 8000000, 0, 0);
		//cv::createTrackbar("Light intensity", null, &this->generator->getImageCreator()->object_intensity, 10E10, 0, 0);
		//	cv::createTrackbar("Object size (mm)", null , &this->generator->getImageCreator()->object_size, 20, 0, 0);   //TODO: allow object size to be adjusted during simulation
		cout << "Virtual Camera From File Controls added" << endl;
		collect_particledata_from_camera = true;
		//to solve the error global window.cpp:702 cv::createTrackbar UI / Trackbar(FrameRateVirtualCamerasFile@Virtual Camera from File Control Window) : Using 'value' pointer is unsafe and deprecated.Use NULL as value pointer.To fetch trackbar value setup callback.
		//// Callback function for the trackbar
		//void onFrameRateChange(int value, void* userdata) {
		//	// Cast userdata to the appropriate type
		//	// For example, if userdata is a pointer to an int:
		//	int* frame_rate_level = static_cast<int*>(userdata);
		//	*frame_rate_level = value; // Update the value
		//	// Perform any additional actions based on the new value
		//}

		//// Usage of cv::createTrackbar with the callback function
		//int frame_rate_level = 0; // Initialize the frame rate level
		//cv::createTrackbar("FrameRateVirtualCamerasFile", this->getWindowName(), nullptr, 2000000, onFrameRateChange, &frame_rate_level);
	}

	void VirtualCamerasFile::VFcreateImage(lpt::ImageFrame& frame) {
		frame.image = shared_objects->image_type.clone();

		int height = shared_objects->image_type.rows;
		int width = shared_objects->image_type.cols;

		for (int p = 0; p < frame.particles.size(); ++p) {
			double x = frame.particles[p]->x;
			double y = frame.particles[p]->y;
			double r, I;

			r = frame.particles[p]->radius;
			I = frame.particles[p]->intensity;

			cv::Point center(x, y); 
			cv::circle(frame.image, center, r, cv::Scalar(I), -1, 8);

			//if (r >= 1) {
			//	//cv::Point center(static_cast<int>(x), static_cast<int>(y));
			//	//cv::circle(frame.image, center, static_cast<int>(r), cv::Scalar(I, I, I), -1, 8);
			//	cv::Point center(std::ceil(x), std::ceil(y));  // Always round down/up
			//	cv::circle(frame.image, center, std::floor(r), cv::Scalar(I), -1, 8);

			//}
			//else {
			//	if (static_cast<int>(y) >= 0 && static_cast<int>(y) < height && static_cast<int>(x) >= 0 && static_cast<int>(x) < width)
			//		frame.image.at<uchar>(static_cast<int>(y), static_cast<int>(x)) = static_cast<unsigned char>(I);
			//}
		}
		//cv::GaussianBlur(frame.image, frame.image, cv::Size(blur_ksize, blur_ksize), 0, 0);
	}

	void SIST_VirtualCamerasFile::shutdown() {
		cout << "\t-------SIST VirtualCameras File Shutdown Complete" << endl;
	}

	void NICT_VirtualCamerasFile::shutdown() {
		for (auto& file : input_txt_files) {
			if (file.is_open()) {
				file.close();
			}
		}
		cout << "\t-------NICT VirtualCameras File Shutdown Complete" << endl;
	}

	void SINTC_VirtualCamerasFile::shutdown() {
		cout << "\t-------SIST VirtualCameras File Shutdown Complete" << endl;
	}

	bool SIST_VirtualCamerasFile::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;
		//cerr << " under grab frame groupfrom file now reading frame# " << frame_index << endl;
		for (int camera_id = 0; camera_id < cameras.size(); camera_id++) {

			frame_group[camera_id].frame_index = frame_index;

			std::string imageFilePath = image_dir + "/image_" + std::to_string(camera_id) + "_" + std::to_string(frame_index) + ".png";
			std::string textFilePath = text_dir + "/p_" + std::to_string(camera_id) + "_" + std::to_string(frame_index) + ".txt";
			//newframe.image = cv::imread(imageFilePath, cv::IMREAD_GRAYSCALE);
			frame_group[camera_id].image = cv::imread(imageFilePath, cv::IMREAD_GRAYSCALE);
			//cv::Mat img = cv::imread(imageFilePath, cv::IMREAD_GRAYSCALE);
			//cv::resize(img, img, cv::Size(100, 100));
			if (frame_group[camera_id].image.empty()) {
				cerr << "Failed to load image from " << imageFilePath << std::endl;
			}
			//frame_group[camera_id].image = img.clone();

			//std::string windowName = "Display Window Frame " + std::to_string(frame_index) + " " + std::to_string(camera_id);
			//cout << windowName << endl;
			//cv::imshow(windowName, frame_group[camera_id].image);

			// Wait for a key press
			//cv::waitKey(0);

			std::ifstream particle_file(textFilePath);
			if (particle_file.is_open()) {
				frame_group[camera_id].particles.clear();
				std::string line;
				while (std::getline(particle_file, line)) {
					std::istringstream iss(line);
					std::string id_str, x_str, y_str, radius_str;
					if (std::getline(iss, id_str, ',') && std::getline(iss, x_str, ',') && std::getline(iss, y_str, ',') && std::getline(iss, radius_str, ',')) {
						int id = std::stoi(id_str.substr(id_str.find(':') + 2));
						float x = std::stof(x_str.substr(x_str.find(':') + 2));
						float y = std::stof(y_str.substr(y_str.find(':') + 2));
						float radius = std::stof(radius_str.substr(radius_str.find(':') + 2));
						double intensity = 255.0;
						lpt::ParticleImage::Ptr newparticle = lpt::ParticleImage::create(id, x, y, radius, intensity);
						frame_group[camera_id].particles.push_back(newparticle);
					}
				}
				particle_file.close();
			}
			else {
				cerr << "Unable to open particle file for reading." << std::endl;
			}

		}

		//lpt::ImageFrame& frame = cameras[0].frames[frame_index];
		//cout << " frame.frame_index is " << frame.frame_index << endl;
		//for (int c = 0; c < cameras.size(); ++c) {
		//	lpt::ImageFrame& frame = cameras[c].frames[frame_index];
		//	//cout << " frame.frame_index is " << frame.frame_index << endl;
		//	frame_group[c].frame_index = frame.frame_index;
		//	frame_group[c].image = frame.image.clone();
		//	frame_group[c].particles = frame.particles;
		//}


		if (frame_index < frame_end) {
			//cout << " frame_index is " << frame_index << endl;
			frame_index += frame_step;
		}
		else {
			frame_index = frame_start;
			cout << " frame index restarted back to " << frame_start << endl;
			system("pause");
		}

		boost::posix_time::microseconds sleeptime(this->frame_rate_level);
		boost::this_thread::sleep(sleeptime);

		return true;
	}

	bool NICT_VirtualCamerasFile::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;
		int num_cameras = cameras.size();
		std::string line;
		
		while (frame_index <= frame_end) {
			bool all_frames_found = true;

			for (int camera_id = 0; camera_id < num_cameras; ++camera_id) {
				input_txt_files[camera_id].clear();
				input_txt_files[camera_id].seekg(input_txt_files_last_positions[camera_id]);
				bool frame_found = false;

				// Loop to find the specific frame or the nearest next frame
				while (std::getline(input_txt_files[camera_id], line) && !frame_found) {
					std::smatch matches;
					if (std::regex_search(line, matches, frame_start_pattern)) {
						int file_frame_index = std::stoi(matches[2].str());
						//std::cout << "cam " << camera_id << " file_frame_index " << file_frame_index << ", expected frame_index " << frame_index << std::endl;

						// If the file frame index is less than the current frame_index, keep reading
						if (file_frame_index < frame_index) {
							//cout << " cotinue reading more " << endl;
							continue;
						}

						// If the file frame index is greater than the target frame index + frame_step, update frame_index to match file_frame_index
						if (file_frame_index > frame_index) {
							std::cout << "Skipping to exiting file frame " << file_frame_index << " as GT frame_index = " << frame_index << endl;
							frame_index = file_frame_index;
						}
						if (file_frame_index == frame_index) {
							//cout << " starting reading frame index " << file_frame_index << " for camera " << camera_id << std::endl;
							// Process the frame when file_frame_index matches the updated frame_index
							frame_group[camera_id].frame_index = file_frame_index;
							frame_group[camera_id].particles.clear();

							// Save the position of the file after reading the frame index in correct location now
							input_txt_files_last_positions[camera_id] = input_txt_files[camera_id].tellg(); // Save position

							// Read particle data for the frame
							while (std::getline(input_txt_files[camera_id], line) && line.substr(0, 4) != "====") {
								std::istringstream iss(line);
								std::string id_str, x_str, y_str, radius_str;
								if (std::getline(iss, id_str, ',') && std::getline(iss, x_str, ',') &&
									std::getline(iss, y_str, ',') && std::getline(iss, radius_str, ',')) {
									int id = std::stoi(id_str.substr(id_str.find(':') + 2));
									float x = std::stof(x_str.substr(x_str.find(':') + 2));
									float y = std::stof(y_str.substr(y_str.find(':') + 2));
									float radius = std::stof(radius_str.substr(radius_str.find(':') + 2));
									double intensity = 255.0; // Assuming intensity is needed
									lpt::ParticleImage::Ptr newparticle = lpt::ParticleImage::create(id, x, y, radius, intensity);
									frame_group[camera_id].particles.push_back(newparticle);
								}
							}

							frame_found = true;
							VFcreateImage(frame_group[camera_id]);
						}
					}
				}

				// If any frame is not found, set all_frames_found to false
				if (!frame_found) {
					cout << "One frame is missing in the dataset, " << frame_index << " for camera " << camera_id << endl;
					all_frames_found = false;
				}
			}

			// Only increment frame_index by frame_step if all frames were successfully found
			if (all_frames_found) {
				frame_index += frame_step;
			}

			// Reset mechanism for loop
			if (frame_index > frame_end) {
				std::cout << "Reached the end of the frame range, restarting." << std::endl;
				system("pause");
				frame_index = frame_start;
				std::for_each(input_txt_files.begin(), input_txt_files.end(), [](std::ifstream& f) { f.seekg(0); });
				std::fill(input_txt_files_last_positions.begin(), input_txt_files_last_positions.end(), std::streampos(0));
			}

			//boost::posix_time::microseconds sleeptime(this->frame_rate_level);
			//boost::this_thread::sleep(sleeptime);
			boost::this_thread::sleep_for(boost::chrono::microseconds(this->frame_rate_level));
			//https://stackoverflow.com/questions/4265310/how-to-sleep-a-c-boost-thread

			// dummy instead of sleep
			//volatile int dummy_counter = 0;
			//for (int i = 0; i < frame_rate_level * 1000; ++i) {
			//	dummy_counter += (i ^ (dummy_counter << 1)) % 100;  // Simple operation to take up time
			//}
			return true; // Processed one frame group successfully
		}
	}

	bool SINTC_VirtualCamerasFile::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
		static int last_camera_id = -1; // To track changes in the camera ID
		static bool end_message_displayed = false; // To ensure the end message is displayed only once

		int num_cameras = shared_objects->cameras.size();
		int camera_id = *current_camera_idx;

		// Reset variables if the camera ID changes
		if (camera_id != last_camera_id) {
			current_blank_count = 0;
			//file_index_map[camera_id] = 0; // this is not needed as the file index is already set to 0 in the initialization
			end_message_displayed = false;
			last_camera_id = camera_id;
		}

		int& file_index = file_index_map[camera_id];

		// Calibrator settings
		if (calibrator && !calibrator_settings) {
			calibrator->setFrameInterval(1);
			calibrator->setIntParamDataCollection(true);
			calibrator_settings = true;
			cout << "Calibrator settings set in virtual mode" << endl;
		}

		// Supply blank images for the first blank_repeat_count frames
		if (current_blank_count < blank_repeat_count) {
			for (int cam_id = 0; cam_id < num_cameras; ++cam_id) {
				frame_group[cam_id].image = shared_objects->image_type.clone();
				frame_group[cam_id].frame_index = current_blank_count;
				frame_group[cam_id].particles.clear();
			}
			//cout << "Blank frame for all cameras frame index " << current_blank_count << endl;
			current_blank_count++;
			boost::posix_time::microseconds sleeptime(this->frame_rate_level);
			boost::this_thread::sleep(sleeptime);
			return true;
		}

		// Read real images
		if (file_index < file_map[camera_id].size()) {
			boost::filesystem::path imageFilePath = file_map[camera_id][file_index];
			file_index++; // Move to the next file index for the next call

			frame_group[camera_id].frame_index = blank_repeat_count + file_index - 1;
			frame_group[camera_id].particles.clear();
			std::string corrected_image_path = imageFilePath.generic_string(); // Ensure correct path format
			//cout << "Reading frame " << frame_group[camera_id].frame_index << " for camera " << camera_id << " from " << corrected_image_path << endl;

			frame_group[camera_id].image = cv::imread(corrected_image_path, cv::IMREAD_GRAYSCALE);
			// Check if the image was successfully loaded
			if (frame_group[camera_id].image.empty()) {
				cout << "Failed to load image for camera " << camera_id << " from file: " << corrected_image_path << std::endl;
				return false; // Return false if the image could not be loaded
			}

			// Ensure other cameras have a blank image
			for (int other_camera_id = 0; other_camera_id < num_cameras; other_camera_id++) {
				if (other_camera_id != camera_id) {
					frame_group[other_camera_id].image = shared_objects->image_type.clone();
					frame_group[other_camera_id].frame_index = blank_repeat_count + file_index - 1;
					frame_group[other_camera_id].particles.clear();
				}
			}
			boost::posix_time::microseconds sleeptime(this->frame_rate_level);
			boost::this_thread::sleep(sleeptime);
			return true;
		}

		// Supply blank images for the final blank_repeat_count frames
		if (file_index >= file_map[camera_id].size() && current_blank_count < 2 * blank_repeat_count) {
			for (int cam_id = 0; cam_id < num_cameras; ++cam_id) {
				frame_group[cam_id].image = shared_objects->image_type.clone();
				frame_group[cam_id].frame_index = blank_repeat_count + file_index - 1;
				frame_group[cam_id].particles.clear();
			}
			//cout << "Blank frame for all cameras frame index " << current_blank_count << endl;
			current_blank_count++;
			boost::posix_time::microseconds sleeptime(this->frame_rate_level);
			boost::this_thread::sleep(sleeptime);
			return true;
		}

		// Display message after all frames are processed, only once
		if (current_blank_count >= 2 * blank_repeat_count && !end_message_displayed) {
			cout << "\t Reached the end of images Click on the calibration button for camera " << camera_id << endl;
			end_message_displayed = true;
		}
		boost::posix_time::microseconds sleeptime(this->frame_rate_level);
		boost::this_thread::sleep(sleeptime);
		return false; // Default return false if nothing was processed
	}

	/***** mahdi : lpt::VirtualCamerasImg class implementation*****/
	bool VirtualCamerasImg::initializeCameras() {
		std::cout << "Initializing virtual cameras from Images" << std::endl;
		auto& cameras = shared_objects->cameras;
		//auto& camera_pairs = shared_objects->camera_pairs;
		shared_objects->camera_type = lpt::VIRTUAL;
		shared_objects->image_type = cv::Mat::zeros(cv::Size(1280, 1024), CV_8UC1);
		shared_objects->half_image_type = cv::Mat::zeros(cv::Size(640, 512), CV_8UC1);
		lpt::readCamerasFile(camerasfile, cameras);
		lpt::ImageFrame init_frame; //setting the beging framegroup an empty one to start the cameras
		//init_frame.image = shared_objects->image_type; init_frame.frame_index = 123;
		init_frame.image = shared_objects->half_image_type; init_frame.frame_index = 123;
		for (int c = 0; c < cameras.size(); c++)
			cameras[c].frames.push_back(init_frame);
		//std::system("pause");
		shared_objects->frame_rate = virtual_fps;//maybe need to modify later
		cout << " virtual_fps = " << virtual_fps << endl;

		if (!cameras[0].frames.empty()) {
			cout << " *************camera status set to true************** " << endl;
			this->setCamerasStatus(true);
			//std::system("pause");
			return true;
		}
		else
			return false;		
	}

	void VirtualCamerasImg::addControls() {
		cv::createTrackbar("FrameRate", this->getWindowName(), &frame_rate_level, 4000000, 0, 0);		
		cout << "Virtual Camera Img Controls added" << endl;
		setParticleCollectionFromCamera(false);
	}

	bool VirtualCamerasImg::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;
		int num_cameras = cameras.size();

		// Supply blank images for the first blank_repeat_count frames
		//if (current_blank_count < blank_repeat_count) {
		//	for (int cam_id = 0; cam_id < num_cameras; ++cam_id) {
		//		frame_group[cam_id].image = shared_objects->half_image_type.clone();
		//		frame_group[cam_id].frame_index = current_blank_count;
		//		frame_group[cam_id].particles.clear();
		//	}
		//	current_blank_count++;
		//	boost::posix_time::microseconds sleeptime(this->frame_rate_level);
		//	boost::this_thread::sleep(sleeptime);
		//	return true;
		//}

		// Read real images
		for (int cam_id = 0; cam_id < num_cameras; ++cam_id) {
			frame_group[cam_id].frame_index = frame_index;
			string imageFilePath = image_dir + "/image_" + to_string(cam_id) + "_" + to_string(frame_index) + ".png";
			if (!boost::filesystem::exists(imageFilePath)) {
				std::cout << "Image file not found: " << imageFilePath << std::endl;
				frame_index += 1; // Move to the next frame
				return false;
			}
			cout << "Reading frame " << frame_index << " for camera " << cam_id << " from " << imageFilePath << endl;
			double alpha = 2.0;  // Contrast control (1.0-3.0 for higher contrast)
			int beta = 50;       // Brightness control (0-100 for increasing brightness)
			cv::Mat file_image = cv::imread(imageFilePath, cv::IMREAD_GRAYSCALE);
			if (file_image.empty()) {
				cout << "Failed to load image from " << imageFilePath << std::endl;
				frame_index += 1;
				return false;
			}
			//file_image.convertTo(file_image, -1, alpha, beta);
			frame_group[cam_id].image = file_image;
				
			frame_group[cam_id].particles.clear();
			if (frame_group[cam_id].image.empty()) {
				cout << "Failed to load image from " << imageFilePath << std::endl;
			}
			frame_group[cam_id].frame_index = frame_index;
		}
		if (frame_index < frame_end) {
			//cout << " frame_index is " << frame_index << endl;
			frame_index += frame_step;
		}
		else {
			frame_index = frame_start;
			cout << " frame index restarted back to " << frame_start << endl;
			system("pause");
		}

		boost::posix_time::microseconds sleeptime(this->frame_rate_level);
		boost::this_thread::sleep(sleeptime);

		return true;
	}

	void VirtualCamerasImg::shutdown() {
		std::cout << " \t Shutting down the virtual camera image system" << std::endl;
	}

	//end mahdi


#ifdef USE_NP_CAMERASDK 
	//---Using Natural Point camera SDK---

	/*****lpt::Optitrack class implementation*****/

	bool Optitrack::initializeCameras() {
		this->shared_objects->camera_type = lpt::OPTITRACK;
		auto& cameras = shared_objects->cameras;
		cout << "Searching for attached cameras" << endl;
		CameraLibrary::CameraManager::X().WaitForInitialization();
		cout << (CameraLibrary::CameraManager::X().AreCamerasInitialized() ?
			"\t --Complete" : "\t --Failed") << endl;
		CameraLibrary::CameraList list;

		if (list.Count() <= 0) {
			cout << "No Cameras detected:  Exiting" << endl;
			return false;
		}
		else {
			optitrack_cameras.resize(list.Count());
			cameras.resize(list.Count());
			cout << "Number of cameras detected: " << optitrack_cameras.size() << endl;
		}
		//CameraLibrary_EnableDevelopment();
		//CameraLibrary::sCameraResolution resolution;
		//resolution.ResolutionID = 1;
		//resolution.Width = 640;
		//resolution.Height = 512;
		//CameraLibrary::
		for (int i = 0; i < optitrack_cameras.size(); i++) {
			std::shared_ptr<CameraLibrary::Camera> newcamera = CameraLibrary::CameraManager::X().GetCamera(list[i].UID());
			int camera_id = newcamera->CameraID() - 1;
			optitrack_cameras[camera_id] = newcamera;
			cameras[camera_id].id = camera_id;
			stringstream cameraname;
			cameraname << list[i].Name();
			cameras[camera_id].name = cameraname.str();
			cameras[camera_id].sensor_size[0] = optitrack_cameras[camera_id]->ImagerWidth();
			cameras[camera_id].sensor_size[1] = optitrack_cameras[camera_id]->ImagerHeight();
			cameras[camera_id].pixel_size[0] = optitrack_cameras[camera_id]->ImagerWidth() / optitrack_cameras[camera_id]->PhysicalPixelWidth();
			cameras[camera_id].pixel_size[1] = optitrack_cameras[camera_id]->ImagerHeight() / optitrack_cameras[camera_id]->PhysicalPixelHeight();

			cout << "Camera " << camera_id << ": " << list[i].Name() << endl;
			if (!optitrack_cameras[camera_id]) {
				cout << "Fail! camera does not exist " << camera_id << endl;
				return false;
			}
			//Print camera capabilities
			//cout << "\t --Filter Switcher: " << (optitrack_cameras[camera_id]->IsFilterSwitchAvailable() ? "Avaliable" : "Not Avaliable") << endl;
			//cout << "\t --MJPEG Mode: " << (optitrack_cameras[camera_id]->IsMJPEGAvailable() ? "Avaliable" : "Not Avaliable") << endl;
			cout << "\t --Default FrameRate = " << optitrack_cameras[camera_id]->ActualFrameRate() << endl;
			//cout << "\t --Grayscale Mode: " << (optitrack_cameras[camera_id]->IsVideoTypeSupported(Core::GrayscaleMode) ? "Avaliable" : "Not Avaliable") << endl;


			//Set some initial camera operating parameters
			optitrack_cameras[camera_id]->SetFrameRate(120);
			optitrack_cameras[camera_id]->SetVideoType(Core::SegmentMode);
			//optitrack_cameras[camera_id]->SetLateDecompression(false);
			//optitrack_cameras[camera_id]->SetLateDecompression(true);
			//optitrack_cameras[camera_id]->SetLateMJPEGDecompression(false); //from barker not avaialble here
			optitrack_cameras[camera_id]->SetMJPEGQuality(100);
			optitrack_cameras[camera_id]->SetAEC(false);
			optitrack_cameras[camera_id]->SetAGC(true);
			optitrack_cameras[camera_id]->SetExposure(5900); // mahdi this is initial exposure was 40 now at 6k
			optitrack_cameras[camera_id]->SetThreshold(60); // mahdi this is segmentation threshold
			optitrack_cameras[camera_id]->SetIntensity(0);   //IR LED intensity if available
			optitrack_cameras[camera_id]->SetTextOverlay(false);
			optitrack_cameras[camera_id]->SetObjectColor(255);
			optitrack_cameras[camera_id]->SetIRFilter(false);			
		}
		//mahdi code for esync but didnt work try in future
		//std::shared_ptr<CameraLibrary::Camera> esync = nullptr;
		//CameraLibrary::HardwareDeviceList deviceList;
		//int deviceIndex = -1;
		//for (int i = 0; i < deviceList.Count(); ++i)
		//{
		//	std::string devName(deviceList[i].Name());
		//	std::string syncName = "eSync";
		//	const size_t index = devName.find(syncName);
		//	if (index != std::string::npos)
		//	{
		//		deviceIndex = i;
		//		break;
		//	}
		//}
		//if (deviceIndex >= 0)
		//{
		//	esync = CameraLibrary::CameraManager::X().GetDevice(deviceList[deviceIndex].UID());
		//}

		//if (esync && !esync->IsSyncAuthority())
		//{
		//	esync = nullptr;
		//}
		sync = CameraLibrary::cModuleSync::Create(); //mahdi refer to frame sync.txt in camera sdk
		//cModuleSync* sync = CameraLibrary::cModuleSync::Create(); //mahdi this was accoring to the document but didn't work because we have already defined sync in the optitrack class like this
		for (int i = 0; i < optitrack_cameras.size(); ++i)
			sync->AddCamera(optitrack_cameras[i]);

		cout << "number of cameras in the sync module: " << sync->CameraCount() << endl;

		sensor_dim = cv::Size(optitrack_cameras[0]->Width(), optitrack_cameras[0]->Height());
		cout << "Camera sensor size [" << sensor_dim.width << " x " << sensor_dim.height << "]" << endl;

		shared_objects->image_type = cv::Mat(sensor_dim, CV_8UC1);

		// initialize the half image type
		shared_objects->half_image_type = cv::Mat(cv::Size((int)(optitrack_cameras[0]->Width() / 2), (int)(optitrack_cameras[0]->Height() / 2)), CV_8UC1);

		optitrack_frames.resize(sync->CameraCount());
		this->setCamerasStatus(true);

		for (int i = 0; i < optitrack_cameras.size(); ++i) {
			optitrack_cameras[i]->Start();
			//Set some initial camera operating parameters
			//optitrack_cameras[i]->SetFrameRate(30);   //original was 30 
			//optitrack_cameras[i]->SetVideoType(Core::ObjectMode);
			//optitrack_cameras[i]->SetLateDecompression(false);
			////optitrack_cameras[camera_id]->SetLateDecompression(true);
			////optitrack_cameras[camera_id]->SetLateMJPEGDecompression(false); //from barker
			//optitrack_cameras[i]->SetMJPEGQuality(100);
			//optitrack_cameras[i]->SetAEC(true);
			//optitrack_cameras[i]->SetAGC(true);
			//optitrack_cameras[i]->SetExposure(5900); 
			//optitrack_cameras[i]->SetThreshold(100);
			//optitrack_cameras[i]->SetIntensity(0);   //IR LED intensity if available
			//optitrack_cameras[i]->SetTextOverlay(false);
			//optitrack_cameras[i]->SetObjectColor(255);
			//optitrack_cameras[i]->SetIRFilter(false);
			//cout << "MaximumThreshold()=" << optitrack_cameras[i]->MaximumThreshold() << endl;
			//cout << "MinimumThreshold()=" << optitrack_cameras[i]->MinimumThreshold() << endl;
			//cout << "MaximumExposureValue()=" << optitrack_cameras[i]->MaximumExposureValue() << endl;
			//cout << "MinimumExposureValue()()=" << optitrack_cameras[i]->MinimumExposureValue() << endl;
			cout << "\t Camera " << i << endl;
			//cout << " Blocking mask enabled: " << (optitrack_cameras[i]->IsBlockingMaskEnabled() ? "yes" : "no") << endl;
			optitrack_cameras[i]->SetEnableBlockingMask(false);
			cout << " Blocking mask enabled: " << (optitrack_cameras[i]->IsBlockingMaskEnabled() ? "yes" : "no") << endl;
			cout << " MJPEGQuality = " << optitrack_cameras[i]->MJPEGQuality() << endl;
			cout << " ActualFrameRate() = " << optitrack_cameras[i]->ActualFrameRate() << endl;
			cout << " FrameRate() = " << optitrack_cameras[i]->FrameRate() << endl;

			//cout << "camera res id " << optitrack_cameras[i]->CameraResolutionID() << endl;
			//cout << "camera res count " << optitrack_cameras[i]->CameraResolutionCount() << endl;
			//CameraLibrary::sCameraResolution res = optitrack_cameras[i]->CameraResolution(0);
			//cout << "camera res width " << res.Width << endl;
			//cout << "camera res height " << res.Height << endl;
			//cout << "camera res id " << res.ResolutionID << endl;
			//cout << "optitrack_cameras[i]->Width()" << optitrack_cameras[i]->Width() << endl;
			//1280x1024 - 640x512
			//cout << "optitrack_cameras [i]" << optitrack_cameras[i]->SerialString() << endl;
			//cout << "optitrack_cameras [i]" << optitrack_cameras[i]->IsFilterSwitchAvailable() << endl;
		}
		shared_objects->frame_rate = optitrack_cameras[0]->ActualFrameRate();
		cout << "\t -- Shared Objects Frame Rate = " << this->getFrameRate() << endl;
		//cout << "\t mahdi  HardwareFrameRate= " << optitrack_cameras[0]->HardwareFrameRate() << endl; //mahdi
		//cout << "\t mahdi  MinimumFrameRateValue= " << optitrack_cameras[0]->MinimumFrameRateValue() << endl; //mahdi
		//cout << "\t mahdi  MaximumFrameRateValue= " << optitrack_cameras[0]->MaximumFrameRateValue() << endl; //mahdi
		//cout << "\t mahdi  MaximumFullImageFrameRateValue= " << optitrack_cameras[0]->MaximumFullImageFrameRateValue() << endl; //mahdi


		//// push back the get particle collection from camera
		//bool getParticleFromCamera = false;
		//for (int i = 0; i < optitrack_cameras.size(); ++i) {
		//	collect_particledata_from_camera_vec.push_back(getParticleFromCamera);
		//	cout << "The " << i << " th" << " camera" << " get particle from camera? " << collect_particledata_from_camera_vec[i] << " " << endl;
		//}
		
		this->startCameraTxtWriters();
		this->startCameraImgWriters();
		return true;
	}

	void Optitrack::shutdown() {

		cout << "------------Shutting Down Cameras------------" << endl;
		optitrack_cameras.clear();
		sync->RemoveAllCameras();
		cModuleSync::Destroy(sync);

		//for (int i = 0; i < optitrack_cameras.size(); ++i) {
		//	optitrack_cameras[i]->Release();
		//	//cout << i << " ";
		//}
		CameraLibrary::CameraManager::X().Shutdown();
		//mahdi joining threads
		cout << " mahdi joining writing threads " << endl;
		stopCameraTxtWriters();
		stopCameraImgWriters();
		//image_writer_thread_group.interrupt_all();
		//image_writer_thread_group.join_all();
		//txtWriter_thread.interrupt();
		//txtWriter_thread.join();

		// start of writing multithreaded txt files
		//// mahdi 5/16/2024 below commented was writing frame_indices vector
		//cout << " opening text file to write indices " << endl;
		//ofstream indices_file("frame_indices.txt");
		//for (const auto& index : frame_indices) {
		//	for (int i=0; i<index.size(); i++)
		//	{
		//		indices_file << "," << index[i];
		//	}
		//	indices_file << endl;
		//}
		//indices_file.close();
		//cout << " indices write success " << endl;
		// mahdi 5/17/2024 below is joining threads for multiple text file writer threads
		//cam0txtwritert.interrupt();
		//cam0txtwritert.join();

		//cam1txtwritert.interrupt();
		//cam1txtwritert.join();

		//cam2txtwritert.interrupt();
		//cam2txtwritert.join();

		//cam3txtwritert.interrupt();
		//cam3txtwritert.join();

		//cam4txtwritert.interrupt();
		//cam4txtwritert.join();

		//cam5txtwritert.interrupt();
		//cam5txtwritert.join();

		//cam6txtwritert.interrupt();
		//cam6txtwritert.join();

		//cam7txtwritert.interrupt();
		//cam7txtwritert.join();
		////end of writing multithreaded txt files
		//cout << " image and text Writer_threads joined " << endl;

		cout << "\t---Optitrack Shutdown Complete, threads joined" << endl;
		system("pause");
	}

	////mahdi 5/16/24 below is the original from simon pc
//bool Optitrack::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
//	CameraLibrary::FrameGroup* native_frame_group = sync->GetFrameGroup();
//	if (native_frame_group) {
//		array < int, 8 > current_indices;
//		fill(current_indices.begin(), current_indices.end(), 0);
//		auto& image_type = shared_objects->image_type;
//		for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id) {
//			optitrack_frames[camera_id] = native_frame_group->GetFrame(camera_id);
//			frame_group[camera_id].image = image_type.clone();
//			optitrack_frames[camera_id]->Rasterize(image_type.size().width, image_type.size().height,
//				static_cast<unsigned int>(image_type.step), 8, frame_group[camera_id].image.data);
//			frame_group[camera_id].frame_index = optitrack_frames[camera_id]->FrameID();
//			frame_group[camera_id].particles.clear();
//			if (WriteRawDataRequested)
//				current_indices[camera_id] = frame_group[camera_id].frame_index;
//			if (this->collect_particledata_from_camera) {
//				//frame_group[camera_id].particles.clear();
//				for (int object_id = 0; object_id < optitrack_frames[camera_id]->ObjectCount(); object_id++) {
//					frame_group[camera_id].particles.push_back
//					(
//						std::move(
//							lpt::ParticleImage::create(
//								object_id,
//								optitrack_frames[camera_id]->Object(object_id)->X(),
//								optitrack_frames[camera_id]->Object(object_id)->Y(),
//								optitrack_frames[camera_id]->Object(object_id)->Radius()
//							)
//						)
//					);
//				}
//			}
//			else {
//				//frame_group[camera_id].image = half_image_type.clone();
//				//optitrack_frames[camera_id]->Rasterize(half_image_type.size().width, half_image_type.size().height, 
//				//	static_cast<unsigned int>(frame_group[camera_id].image.step), 8, frame_group[camera_id].image.data);
//				////////////////////////////////////////////////////////////////////////////////
//				cv::Mat temp = image_type.clone();
//				optitrack_frames[camera_id]->Rasterize(image_type.size().width, image_type.size().height,
//					static_cast<unsigned int>(temp.step), 8, temp.data);
//				cv::Rect ROI(0, 0, image_type.size().width / 2, image_type.size().height / 2);
//				cv::resize(temp(ROI), frame_group[camera_id].image, image_type.size(), 0.0, 0.0, cv::INTER_CUBIC);
//			}
//			optitrack_frames[camera_id]->Release();
//		}
//		if (WriteRawDataRequested)
//			frame_indices.push_back(current_indices);
//		native_frame_group->Release();
//		if (sync->LastFrameGroupMode() != CameraLibrary::FrameGroup::Hardware) {
//			for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id)
//				frame_group[camera_id].image = image_type.clone();
//			cout << "\t Cameras NOT Synchronized: Frame # = " << frame_count << endl;
//		}
//		++frame_count;
//		return true;
//	}
//	else
//		return false;
//}


	bool Optitrack::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
		std::shared_ptr<const CameraLibrary::FrameGroup> native_frame_group = sync->GetFrameGroup();
		if (native_frame_group) {
			int frame_skip_interval = static_cast<int>(min_cam_fps / shared_objects->frame_rate);
			//cout << " frame_skip_interval= " << frame_skip_interval << endl;		
			/* below were the code for fps smaller than 30 that I disabled. mahdi 5/20/25
			if (shared_objects->frame_rate < min_cam_fps && shared_objects->frame_rate > 0) {
				if (frame_count % frame_skip_interval == 0) {
					for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id) {
						this->processCameraData(camera_id, native_frame_group, frame_group);
					}

					if (sync->LastFrameGroupMode() != CameraLibrary::FrameGroup::Hardware) { // mahdi here it checks if synchronization is achieved by hw
						for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id)
							frame_group[camera_id].image = shared_objects->image_type.clone();
						cout << "\t Cameras NOT Synchronized: Frame # = " << frame_count << endl;
					}
					//cout << " mahdi under grab frame gp main loop frame_count= " << frame_count << endl; //mahdi

					//native_frame_group->Release();
					++frame_count;
					return true;
				}
				else {
					//native_frame_group->Release();
					++frame_count;
					return false;
				}
			}
			else {// If target FPS is equal to or greater than the min camera's FPS, process every frame usual
			*/
				for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id) {
					this->processCameraData(camera_id, native_frame_group, frame_group);
				}

				if (sync->LastFrameGroupMode() != CameraLibrary::FrameGroup::Hardware) {
					for (int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id)
						frame_group[camera_id].image = shared_objects->image_type.clone();
					cout << "\t Cameras NOT Synchronized: Frame # = " << frame_count << endl;
				}
				//cout << " mahdi under grab frame gp main loop frame_count= " << frame_count << endl; //mahdi

				//native_frame_group->Release();
				++frame_count;
				return true;
			//}
		}
		else {
			return false; // Return false if no frame group was obtained
		}
	}

	void Optitrack::writeImgFileGeneral(int cameraIndex) {
		std::pair<std::string, cv::Mat> item;
		std::vector<std::pair<std::string, cv::Mat>> batch;
		const int batchSize = 10;  // Adjust as needed

		while (true) {
			camImgWriteQueues[cameraIndex]->wait_and_pop(item);
			batch.push_back(item);

			if (batch.size() >= batchSize) {
				for (const auto& imgItem : batch) {
					const std::string& filename = imgItem.first;
					const cv::Mat& image = imgItem.second;
					bool writeSuccess = cv::imwrite(filename, image);
					if (!writeSuccess) {
						std::cerr << "Failed to write image file: " << filename << std::endl;
					}
				}
				batch.clear();
			}

			boost::this_thread::interruption_point();  // allows thread to be stopped cleanly
		}
	}

	//void Optitrack::writeTxtFiles() {//all together
	//	//boost::thread::id this_id = boost::this_thread::get_id();
	//	//std::cout << " mahdi text_writer_thread thread Id: " << this_id << std::endl;
	//	//std::thread::id this_id1 = std::this_thread::get_id();
	//	//std::cout << " std Current thread Id: " << this_id1 << std::endl; // mahdi both return the same 625c, or 25180
	//	//boost::posix_time::microseconds sleeptime(100);
	//	std::ofstream outputFile("txto/combined_output.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) { // You'll need an exit condition here
	//		//while (this->areCamerasRunning() || !txtWriteQueue.empty()) { // You'll need an exit condition here

	//		txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		//old version with try pop and sleep below
	//		//if (txtWriteQueue.try_pop(item)) {
	//		//	const std::string& sectionHeader = item.first;
	//		//	const std::vector<std::string>& textData = item.second;

	//		//	// Write the filename or section header
	//		//	outputFile << "==== " << sectionHeader << " ====" << std::endl;
	//		//	// Write the actual text data
	//		//	for (const auto& line : textData) {
	//		//		outputFile << line << std::endl;
	//		//	}
	//		//	// Optionally add a separator after each section
	//		//	//outputFile << std::endl << "---- End of " << sectionHeader << " ----" << std::endl;
	//		//}
	//		//else {
	//		//	// Consider adding a delay or exit condition if the queue is empty
	//		//	//boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
	//		//	boost::this_thread::sleep(sleeptime);
	//		//}
	//		boost::this_thread::interruption_point();
	//	}

	//	outputFile.close();
	//}
	// 
	//mahdi writing a generic function for automatic number of cameras with batch
	
	void Optitrack::writeTxtFileGeneral(int cameraIndex) {
		std::string filePath = shared_objects->output_path + "txto/cam" + std::to_string(cameraIndex) + ".txt";
		std::ofstream outputFile(filePath, std::ios_base::app); //append mode
		if (!outputFile.is_open()) {
			std::cout << "Unable to open output file for writing." << std::endl;
			return;
		}
		else {
			std::cout << "Writing cam " << cameraIndex << " at " << filePath << std::endl;
		}

		std::pair<std::string, std::vector<std::string>> item;
		std::vector<std::pair<std::string, std::vector<std::string>>> batch; // Batch storage
		const int batchSize = 10; // Adjust based on memory and performance requirements

		while (true) {
			camTxtWriteQueues[cameraIndex]->wait_and_pop(item);
			batch.push_back(item);

			// If batch size is met, write all items in the batch at once
			if (batch.size() >= batchSize) {
				for (const auto& batchItem : batch) {
					outputFile << "==== " << batchItem.first << " ====" << std::endl;
					for (const auto& line : batchItem.second) {
						outputFile << line << std::endl;
					}
				}
				batch.clear(); // Clear the batch after writing
			}
			boost::this_thread::interruption_point();
		}

		// Write any remaining items in the batch at the end. doesnt run because of interuption.
		for (const auto& batchItem : batch) {
			outputFile << "==== " << batchItem.first << " ====" << std::endl;
			for (const auto& line : batchItem.second) {
				outputFile << line << std::endl;
			}
		}

		outputFile.close();
	}

	void Optitrack::startCameraTxtWriters() {
		int numCameras = optitrack_cameras.size();

		for (int i = 0; i < numCameras; ++i) {
			camTxtWriteQueues.push_back(std::make_unique<lpt::concurrent_queue<std::pair<std::string, std::vector<std::string>>> >());
			camTxtWriteQueues[i]->setCapacity(txt_queue_capacity); 
			camTxtWriterThreads.create_thread(boost::bind(&Optitrack::writeTxtFileGeneral, this, i)); // Launch each camera writer thread			
		}
		cout << "All camera text writer threads have been started." << endl;
		cout << "All txt queues have been set to capacity " << txt_queue_capacity << endl;
	}

	void Optitrack::startCameraImgWriters() {
		int numCameras = optitrack_cameras.size();

		for (int i = 0; i < numCameras; ++i) {
			camImgWriteQueues.push_back(std::make_unique<lpt::concurrent_queue<std::pair<std::string, cv::Mat>> >());
			camImgWriteQueues[i]->setCapacity(img_queue_capacity);
			camImgWriterThreads.create_thread(boost::bind(&Optitrack::writeImgFileGeneral, this, i)); // Launch each camera writer thread			
		}
		cout << "All camera image writer threads have been started." << endl;
		cout << "All image queues have been set to capacity " << txt_queue_capacity << endl;
	}

	void Optitrack::stopCameraTxtWriters() {
		camTxtWriterThreads.interrupt_all();
		camTxtWriterThreads.join_all();
		cout << "All camera text writer threads have been stopped." << endl;
	}

	void Optitrack::stopCameraImgWriters() {
		camImgWriterThreads.interrupt_all();
		camImgWriterThreads.join_all();
		cout << "All camera image writer threads have been stopped." << endl;
	}

	//// start of writing multiple text files manually from 0 to 7
	//void Optitrack::writeTxtFiles0() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam0.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam0 at " << shared_objects->output_path + "txto/cam0.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam0txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles1() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam1.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam1 at " << shared_objects->output_path + "txto/cam1.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam1txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles2() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam2.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam2 at " << shared_objects->output_path + "txto/cam2.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam2txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles3() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam3.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam3 at " << shared_objects->output_path + "txto/cam3.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam3txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles4() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam4.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam4 at " << shared_objects->output_path + "txto/cam4.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam4txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles5() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam5.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam5 at " << shared_objects->output_path + "txto/cam5.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam5txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles6() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam6.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam6 at " << shared_objects->output_path + "txto/cam6.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam6txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}

	//void Optitrack::writeTxtFiles7() {
	//	std::ofstream outputFile(shared_objects->output_path + "txto/cam7.txt");
	//	if (!outputFile.is_open()) {
	//		std::cout << "Unable to open output file for writing." << std::endl;
	//		return;
	//	}
	//	else
	//		cout << " writing cam7 at " << shared_objects->output_path + "txto/cam7.txt" << endl;
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) {
	//		cam7txtWriteQueue.wait_and_pop(item);
	//		const std::string& sectionHeader = item.first;
	//		const std::vector<std::string>& textData = item.second;

	//		// Write the filename or section header
	//		outputFile << "==== " << sectionHeader << " ====" << std::endl;

	//		// Write the actual text data
	//		for (const auto& line : textData) {
	//			outputFile << line << std::endl;
	//		}
	//		boost::this_thread::interruption_point();
	//	}
	//	outputFile.close();
	//}
	//// end of writing multiple text files

	// the function below writes multiple text files
	//void Optitrack::writeTxtFiles() {
	//	std::pair<std::string, std::vector<std::string>> item;
	//	while (true) { // Replace with a proper exit condition
	//		if (!txtWriteQueue.try_pop(item)) {
	//			boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
	//			continue;
	//		}
	//		const std::string& filePath = item.first;
	//		const std::vector<std::string>& lines = item.second;

	//		std::ofstream outputFile(filePath);
	//		if (outputFile.is_open()) {
	//			for (const auto& line : lines) {
	//				outputFile << line << std::endl; // Write each line in the vector to the file
	//			}
	//			outputFile.close();
	//		}
	//		else {
	//			std::cerr << "Unable to open file for writing: " << filePath << std::endl;
	//		}
	//	}
	//}

	////mahdi: not using pgm any more but keeping it for future. commented on 5/6/2024
	//bool Optitrack::writePGM(const std::string& filename, const cv::Mat& image) {
	//	std::ofstream outFile(filename, std::ios::out | std::ios::binary);
	//	if (!outFile.is_open()) {
	//		std::cout << "Error: Unable to open file for writing: " << filename << std::endl;
	//		return false;
	//	}

	//	int width = image.cols;
	//	int height = image.rows;
	//	int maxVal = 255;

	//	outFile << "P5\n" << width << " " << height << "\n" << maxVal << "\n";
	//	outFile.write(reinterpret_cast<const char*>(image.data), image.cols * image.rows * image.elemSize());

	//	if (!outFile.good()) {
	//		std::cerr << "Error: Problem occurred while writing image data to file: " << filename << std::endl;
	//		outFile.close();
	//		return false;
	//	}

	//	outFile.close();
	//	return true;
	//}



	//mahdi changes adding some function for writing data below
	// Process data for a single camera
	void Optitrack::processCameraData(int camera_id, const std::shared_ptr<const CameraLibrary::FrameGroup>& native_frame_group, lpt::ImageFrameGroup& frame_group) {
		std::shared_ptr<const CameraLibrary::Frame> optitrack_frame = native_frame_group->GetFrame(camera_id);
		frame_group[camera_id].frame_index = optitrack_frame->FrameID();
		frame_group[camera_id].particles.clear();
		frame_group[camera_id].image = shared_objects->image_type.clone();
		vector<string> particlesData;
		//if (!optitrack_cameras[camera_id]->IsCameraTempValid()) {
		//	cout << "Camera " << camera_id << " temperature is not valid" << endl;
		//}
		// Initialize your object filter
		//cFrameFilter frameFilter(5, 0.50, 0.50, 0.500); // Initialize with example parameters		
		//CameraLibrary::cLabelObjectFilter objectFilter;
		//CameraLibrary::cSegment* segments() 
		// Initialize the object evaluator with desired filter settings
		//cObjectEval objectEvaluator(10, 0.9, 0.6, 1.0); // Example filter parameters
		//CameraLibrary::cLabelObjects calbelobh;
		//CameraLibrary::cSegment segment1;
		//cout << " under processCameraData maxdimeter " << maxParticleDiameter << endl;
		//cout << " under processCameraData mindimeter " << minParticleDiameter << endl;
		if (collect_particledata_from_camera) {
			optitrack_frame->Rasterize(*optitrack_cameras[camera_id], shared_objects->image_type.size().width, shared_objects->image_type.size().height,
				static_cast<unsigned int>(shared_objects->image_type.step), 8, frame_group[camera_id].image.data);
			//cv::cvtColor(frame_group[camera_id].image, frame_group[camera_id].image, cv::COLOR_GRAY2BGR);
			for (int object_id = 0; object_id < optitrack_frame->ObjectCount(); object_id++) {
				//auto object = optitrack_frame->Object(object_id);
				//const CameraLibrary::cObject* object = optitrack_frame->Object(object_id);
				//cout << " object area = " << object->Area() << endl;
				//double obj_radius = object->Radius();

				//cObjectEval objectEvaluator(10, 0.9, 0.6, 1.0);  // Example parameters: areaMin, aspectRatioLimit, roundnessMinLimit, roundnessMaxLimit
				//CameraLibrary::cObject object 1 = 20;
				//CameraLibrary::cSegment* segment = object->Segments();
				//CameraLibrary::cSegment* segment = const_cast<CameraLibrary::cObject*>(object)->Segments();

				//if (obj_radius >= minParticleDiameter && obj_radius <= maxParticleDiameter) {
				frame_group[camera_id].particles.push_back(
					std::move(lpt::ParticleImage::create(
						object_id,
						optitrack_frame->Object(object_id)->X(),
						optitrack_frame->Object(object_id)->Y(),
						optitrack_frame->Object(object_id)->Radius()
					))
				);
				//cv::circle(frame_group[camera_id].image, cv::Point(optitrack_frame->Object(object_id)->X(), optitrack_frame->Object(object_id)->Y()), optitrack_frame->Object(object_id)->Radius(), cv::Scalar(0, 255, 0), 2);
				if (WriteRawDataRequested) {					
					particlesData.push_back("ID: " + std::to_string(object_id) + ", X: " + std::to_string(optitrack_frame->Object(object_id)->X()) +
						", Y: " + std::to_string(optitrack_frame->Object(object_id)->Y()) + ", R: " + std::to_string(optitrack_frame->Object(object_id)->Radius()));
				}
			}
			//}
			if (WriteRawDataRequested) {
				//auto now = std::chrono::system_clock::now();
				//auto now_c = std::chrono::system_clock::to_time_t(now);
				//std::tm now_tm;// = *std::localtime(&now_c);
				//localtime_s(&now_tm, &now_c);
				//std::ostringstream timestampStream;
				//timestampStream << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S");
				//std::string timestamp = timestampStream.str();
				std::string textFilePath = "p_" + std::to_string(camera_id) + "_" + std::to_string(frame_group[camera_id].frame_index) + ".txt";
				std::string imageFilePath = shared_objects->output_path + "imgo/image_" + std::to_string(camera_id) + "_" + std::to_string(frame_group[camera_id].frame_index) + ".png";

				//if (!txtWriteQueue.full()) {
				//	txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
				//	//std::cout << "particle data added to txt write queue for camera " << camera_id << std::endl;
				//}
				//else {
				//	std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
				//}				

				//if (!imgWriteQueue.full() && camera_id == 1) {
				//	// Making a deep copy of the image to ensure safe multi-threaded operation
				//	cv::Mat imageCopy = frame_group[camera_id].image.clone();
				//	imgWriteQueue.push(std::make_pair(imageFilePath, imageCopy));
				//	//std::cout << "Image data added to img write queue for camera " << camera_id << std::endl;
				//}
				//else {
				//	//std::cout << "img write queue is full, skipping frame for camera " << camera_id << std::endl;
				//}
				// start of writing multithreaded txt files
				if (!camTxtWriteQueues[camera_id]->full()) {
					camTxtWriteQueues[camera_id]->push(std::make_pair(textFilePath, particlesData));
					camImgWriteQueues[camera_id]->push(std::make_pair(imageFilePath, frame_group[camera_id].image));
				}
				else {
					std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
				}
				/*switch (camera_id) {
				case 0:
					if (!cam0txtWriteQueue.full()) {
						cam0txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 1:
					if (!cam1txtWriteQueue.full()) {
						cam1txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 2:
					if (!cam2txtWriteQueue.full()) {
						cam2txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 3:
					if (!cam3txtWriteQueue.full()) {
						cam3txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 4:
					if (!cam4txtWriteQueue.full()) {
						cam4txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 5:
					if (!cam5txtWriteQueue.full()) {
						cam5txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 6:
					if (!cam6txtWriteQueue.full()) {
						cam6txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				case 7:
					if (!cam7txtWriteQueue.full()) {
						cam7txtWriteQueue.push(std::make_pair(textFilePath, particlesData));
					}
					else {
						std::cout << "txt write queue is full, skipping frame for camera " << camera_id << std::endl;
					}
					break;
				default:
					std::cerr << "Invalid camera_id: " << camera_id << std::endl;
				}*/
				// end of writing multithreaded txt files
				particlesData.clear();
			}
		}
		else {
			//mahdi: Yu has used a temp to define half_size image for video mode, but I replaced with another command half_image_type
			cv::Mat temp = shared_objects->image_type.clone();
			optitrack_frame->Rasterize(*optitrack_cameras[camera_id], shared_objects->image_type.size().width, shared_objects->image_type.size().height,
				static_cast<unsigned int>(temp.step), 8, temp.data);
			cv::Rect ROI(0, 0, shared_objects->image_type.size().width / 2, shared_objects->image_type.size().height / 2);
			cv::resize(temp(ROI), frame_group[camera_id].image, shared_objects->image_type.size(), 0.0, 0.0, cv::INTER_CUBIC);
			//frame_group[camera_id].image = shared_objects->half_image_type.clone();
			//optitrack_frame->Rasterize(*optitrack_cameras[camera_id], shared_objects->half_image_type.size().width, shared_objects->half_image_type.size().height,
			//	static_cast<unsigned int>(shared_objects->half_image_type.step), 8, frame_group[camera_id].image.data);
			//cv::GaussianBlur(frame_group[camera_id].image, frame_group[camera_id].image, cv::Size(5, 5), 1, 1);
			if (WriteRawDataRequested) {
				//auto now = std::chrono::system_clock::now();
				//auto now_c = std::chrono::system_clock::to_time_t(now);
				//std::tm now_tm = *std::localtime(&now_c);
				//std::ostringstream timestampStream;
				//timestampStream << std::put_time(&now_tm, "%H-%M-%S");
				//std::string timestamp = timestampStream.str();
				std::string imageFilePath = shared_objects->output_path + "imgo/image_" + std::to_string(camera_id) + "_" + std::to_string(frame_group[camera_id].frame_index) + ".png";

				if (!camImgWriteQueues[camera_id]->full()) {
					cv::Mat imageCopy = frame_group[camera_id].image.clone();
					camImgWriteQueues[camera_id]->push(std::make_pair(imageFilePath, imageCopy));
				}
				else {
					std::cout << "img write queue is full, skipping frame for camera " << camera_id << std::endl;
				}

				// below is the one common queue for all cameras
				//if (!imgWriteQueue.full()) {
				//	// Making a deep copy of the image to ensure safe multi-threaded operation
				//	cv::Mat imageCopy = frame_group[camera_id].image.clone();
				//	imgWriteQueue.push(std::make_pair(imageFilePath, imageCopy));
				//	//std::cout << "Image data added to img write queue for camera " << camera_id << std::endl;
				//}
				//else {
				//	std::cout << "img write queue is full, skipping frame for camera " << camera_id << std::endl;
				//}
			}
		}
		//optitrack_frame->Release();
	}

	//bool Optitrack::grabFrameGroup(lpt::ImageFrameGroup& frame_group) {
	//	//this is the count for print, delete it later
	//	//int countt = 0;
	//
	//	CameraLibrary::FrameGroup* native_frame_group = sync->GetFrameGroup();
	//	//here try to cout the frame ID to see if there is actural frame rate
	//	//cout<<"new frame id is: "<<native_frame_group->TimeStamp()<<endl;
	//	/*
	//	SYSTEMTIME time;
	//	GetSystemTime(&time);
	//	LONG time_ms = (time.wSecond * 1000) + time.wMilliseconds;
	//	cout<<"system time is: "<<time_ms<<endl;
	//	*/
	//	//countt += 1;
	//	if(native_frame_group && native_frame_group->Count() == shared_objects->cameras.size()) {
	//		auto& image_type = shared_objects->image_type;
	//		auto& half_image_type = shared_objects->half_image_type;
	//
	//		for ( int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id ) { 
	//			optitrack_frames[camera_id] = native_frame_group->GetFrame(camera_id);
	//
	//			//frame_group[camera_id].image = image_type.clone();  // move inside the if statements
	//
	//			frame_group[camera_id].frame_index = optitrack_frames[camera_id]->FrameID();
	//			
	//			//here to test under different video mode the frame time interval
	//			int frame_id = frame_group[camera_id].frame_index;
	//			/*
	//			if(optitrack_frames[camera_id]->IsHardwareTimeStamp() && camera_id == 0 && frame_id % 100 == 0){
	//				//cout<<"  ---frame_index is: ---"<<frame_group[camera_id].frame_index<<endl;
	//				
	//				cout<<"\n---there is HardwareTimeStamp---"<<endl;
	//				cout<<"   HardwareTimeStamp is: "<<optitrack_frames[camera_id]->HardwareTimeStamp()<<endl;
	//				cout<<"   HardwareTimeFreq  is: "<<optitrack_frames[camera_id]->HardwareTimeFreq()<<endl;
	//				cout<<"   TimeStame         is: "<<optitrack_frames[camera_id]->TimeStamp()<<endl;
	//				cout<<"   MJPEGQuality      is: "<<optitrack_frames[camera_id]->MJPEGQuality()<<endl;
	//				cout<<"   JPEGImageSize     is: "<<optitrack_frames[camera_id]->JPEGImageSize()<<endl;
	//				
	//			}
	//			*/
	//			if (this->collect_particledata_from_camera_vec[camera_id]) { // This line is for single camera system that might have different video mode from others
	//			//if( this->collect_particledata_from_camera ) { // This line is for uniform camera system
	//				frame_group[camera_id].image = image_type.clone();  // was on top before
	//
	//				frame_group[camera_id].particles.clear();
	//				for (int object_id = 0; object_id < optitrack_frames[camera_id]->ObjectCount(); object_id++) {
	//					auto object = optitrack_frames[camera_id]->Object(object_id);
	//
	//					/*
	//					//here to check the object
	//					if(camera_id == 0 && frame_id % 100 == 0 && object_id == 0){
	//						//cout<<"  ---frame_index is: ---"<<frame_group[camera_id].frame_index<<endl;
	//				
	//						cout<<"\n---there is object information---"<<endl;
	//						cout<<"   left, right, top bottom are: "<<object->Left()<<" "<<object->Right()<<" "<<object->Top()<<" "<<object->Bottom()<<endl;
	//						cout<<"   Area  is:                    "<<object->Area()<<endl;
	//						cout<<"   Radius			       is: "<<object->Radius()<<endl;
	//						cout<<"   Aspect                   is: "<<object->Aspect()<<endl;
	//						cout<<"   Roundness                is: "<<object->Roundness()<<endl;
	//						cout<<"   Weight and height are:       "<<object->Width()<<" "<<object->Height()<<endl;
	//						cout<<"   X and y                  are "<<object->X()<<" "<<object->Y()<<endl;
	//					}
	//					*/
	//					frame_group[camera_id].particles.push_back
	//						(
	//							std::move( 
	//								//lpt::ParticleImage::create( object_id, object->X(), object->Y(), object->Radius() ) 
	//								lpt::ParticleImage::create(object_id, object->X(), object->Y(), object->Radius(), object->Area(), object->Roundness())
	//							) 
	//						);
	//				}
	//			}
	//			else {
	//				//frame_group[camera_id].image = half_image_type.clone();
	//
	//				//optitrack_frames[camera_id]->Rasterize(half_image_type.size().width, half_image_type.size().height, 
	//				//	static_cast<unsigned int>(frame_group[camera_id].image.step), 8, frame_group[camera_id].image.data);
	//
	//				////////////////////////////////////////////////////////////////////////////////
	//				cv::Mat temp = image_type.clone();
	//
	//				//this write is used to be stored
	//				//cv::Mat temp_write = image_type.clone();
	//				//this line need to be deleted, just to verify the intensity of rasterize
	//				
	//
	//				optitrack_frames[camera_id]->Rasterize(image_type.size().width, image_type.size().height, 
	//					static_cast<unsigned int>(temp.step), 8, temp.data);
	//
	//				cv::Rect ROI(0, 0, image_type.size().width/2, image_type.size().height/2);
	//				
	//
	//				/*
	//				//following is used to create temp_write
	//				//cv::resize(temp(ROI), temp_write, image_type.size());
	//				string output_path = "C:\\LPT\\LPT-fork\\data\\output\\images\\";
	//				
	//				string camera_id_str = to_string(static_cast<long long>(camera_id)) + "\\";
	//				string frame_id_str = to_string(static_cast<long long>(frame_id)) + ".jpg";
	//				output_path = output_path + camera_id_str + frame_id_str;
	//				//cout<<"\n output path is: "<<output_path<<endl;
	//				if(frame_id >= 1999 && frame_id <= 2100){
	//					cv::imwrite(output_path, temp);
	//					if(frame_id % 100 == 0)
	//						cout<<"\n-----succeed to write image"<<endl;
	//				}
	//				*/
	//				cv::resize(temp(ROI), frame_group[camera_id].image, image_type.size(), 0.0, 0.0, cv::INTER_CUBIC);
	//
	//			}
	//			optitrack_frames[camera_id]->Release();
	//		}
	//		native_frame_group->Release();
	//		if(sync->LastFrameGroupMode() != CameraLibrary::FrameGroup::Hardware) {
	//			for ( int camera_id = 0; camera_id < native_frame_group->Count(); ++camera_id ) 
	//				frame_group[camera_id].image = image_type.clone();
	//			cout << "\t Cameras NOT Synchronized: Frame # = " << frame_count << endl;  
	//		}
	//		++frame_count;
	//		return true;
	//	} else 
	//		return false;
	//	
	//}

	void Optitrack::addControls() {
		void* optitrack_void_ptr = static_cast<void*> (this);
		void* cameras_void_ptr = static_cast<void*> (&optitrack_cameras);

		string null;
		//auto camera = optitrack_cameras[0];
		init_video_mode = 1;		// {0 = Precision, 1 = Segment, 2 = Object, 3 = MJPEG Mode}
		init_threshold = 60;
		init_exposure = 5900; // mahdi from 40 to 5900 
		init_framerate_value = 120; //mahdi inputting variable framerate, not fixed mode

		// mahdi 6/17/2024 below are old and not needed anymore
		//int max_threshold = camera->MaximumThreshold();
		//int min_threshold = camera->MinimumThreshold();		
		//int max_exposure = camera->MaximumExposureValue();
		//int min_exposure = camera->MinimumExposureValue();
		//init_intensity = 5;
		//init_framerate_mode = 0;    //Mode number: {2 = 100%, 1 = 50%, 0 = 25%} for V120:SLIM cameras 
		init_camera_idx = 0;

		cv::createButton("IR Filter", callbackSetIRFilter, cameras_void_ptr, cv::QT_CHECKBOX, 1); //mahdi initial was 1
		cv::createButton("Automatic Gain Control", callbackSetAGC, cameras_void_ptr, cv::QT_CHECKBOX, 1);
		cv::createButton("Automatic Exposure Control", callbackSetAEC, cameras_void_ptr, cv::QT_CHECKBOX, 0);
		cv::createButton("Text Overlay", callbackSetTextOverlay, cameras_void_ptr, cv::QT_CHECKBOX, 0);
		cv::createButton("Write Raw Data", callbackSetWritingRawData, optitrack_void_ptr, cv::QT_CHECKBOX, 0); //mahdi for raw data writing added
		cv::createTrackbar("VideoMode", null, &init_video_mode, 4, callbackSetVideoType, optitrack_void_ptr);
		cv::createTrackbar("FrameRateValue", null, &init_framerate_value, 240, callbackSetFrameRate, optitrack_void_ptr);
		cv::createTrackbar("Threshold All", null, &init_threshold, 50, callbackSetThreshold, cameras_void_ptr);
		cv::createTrackbar("Threshold Current", null, &init_threshold, 50, callbackSetCurThreshold, optitrack_void_ptr);
		cv::createTrackbar("Exposure All", null, &init_exposure, 80000, callbackSetExposure, cameras_void_ptr);
		cv::createTrackbar("Exposure Current", null, &init_exposure, 80000, callbackSetCurExposure, optitrack_void_ptr);
		//cv::createTrackbar("VideoMode", null , &init_video_mode, 3, callbackSetVideoType, optitrack_void_ptr);//commented because of test of grayscale				
		//cv::createTrackbar("FrameRateMode", null , &init_framerate_mode, 2, callbackSetFrameRate, optitrack_void_ptr);//this is the original track bar with 3 frameRates
	}

	//-------------OPTITRACK CALL BACK FUNCTIONS ------------------------------

	//void callbackSetCurrentVideoType(int mode, void* data) {
	//	Optitrack* system = static_cast< Optitrack*> (data);
	//	vector<CameraLibrary::Camera*>& optitrack_cameras = system->optitrack_cameras;
	//	Core::eVideoMode new_mode;
	//
	//	cout << "Setting Camera Mode of " << system->current_camera_idx << " ";
	//	switch (mode) {
	//	case 0:
	//		cout << "Precision" << endl;
	//		new_mode = Core::PrecisionMode;
	//		system->setCurCamParCol(true);
	//		break;
	//	case 1:
	//		cout << "Segment" << endl;
	//		new_mode = Core::SegmentMode;
	//		system->setCurCamParCol(true);
	//		break;
	//	case 2:
	//		cout << "Object" << endl;
	//		new_mode = Core::ObjectMode;
	//		system->setCurCamParCol(true);
	//		break;
	//	case 3:
	//		cout << "MJPEG" << endl;
	//		new_mode = Core::MJPEGMode;
	//		system->setCurCamParCol(false);
	//		break;
	//		// case for grayscale mode
	//	case 4:
	//		cout << "Grayscale" << endl;
	//		new_mode = Core::GrayscaleMode;
	//		system->setCurCamParCol(false);
	//		break;
	//	default:
	//		new_mode = Core::ObjectMode;
	//		break;
	//	}
	//
	//	if (new_mode == Core::MJPEGMode && optitrack_cameras[system->current_camera_idx]->IsMJPEGAvailable() == false)
	//		cout << "Camera " << system->current_camera_idx << " does not support MJPEG Mode" << endl;
	//	else {
	//		optitrack_cameras[system->current_camera_idx]->SetVideoType(new_mode);
	//	}
	//
	//}
	//
	//void callbackSetCurrentExpTime(int value, void* data){
	//	Optitrack* system = static_cast< Optitrack*> (data);
	//	vector<CameraLibrary::Camera*>& optitrack_cameras = system->optitrack_cameras;
	//
	//	value += optitrack_cameras[0]->MinimumExposureValue();
	//	cout << "Setting Exposure of " <<system->current_camera_idx << " th camera with exposure = " << value;
	//
	//	optitrack_cameras[system->current_camera_idx]->SetExposure(value);
	//	Sleep(2);
	//}

	void callbackSetVideoType(int mode, void* data)
	{
		Optitrack* camera_system = static_cast<Optitrack*> (data);
		vector<std::shared_ptr<CameraLibrary::Camera>>& optitrack_cameras = camera_system->optitrack_cameras;
		Core::eVideoMode new_mode;
		cout << "Setting Camera Mode: ";
		switch (mode) {
		case 0: cout << "Precision"; new_mode = Core::PrecisionMode; break;
		case 1: cout << "Segment"; new_mode = Core::SegmentMode; break;
		case 2: cout << "Object"; new_mode = Core::ObjectMode; break;
		case 3: cout << "MJPEG"; new_mode = Core::MJPEGMode; break;
		case 4: cout << "Grayscale"; new_mode = Core::GrayscaleMode; break;
		default:
			cout << "Precision Mode" << endl;
			new_mode = Core::PrecisionMode;
			break;
		}
		camera_system->setParticleCollectionFromCamera(new_mode != Core::MJPEGMode);
		for (int id = 0; id < optitrack_cameras.size(); id++) {
			if (optitrack_cameras[id]->IsVideoTypeSupported(new_mode)) {
				optitrack_cameras[id]->SetVideoType(new_mode);
				cout << " , " << mode;
			}
			else {
				cout << "Camera " << id << ": Video type " << mode << " not supported." << endl;
			}
		}
		cout << endl;
	}

	void callbackSetCurExposure(int value, void* data) {
		lpt::Optitrack* camera_system = static_cast<lpt::Optitrack*> (data);
		vector<std::shared_ptr<CameraLibrary::Camera>> optitrack_cameras = camera_system->getOptitrackCameras();
		optitrack_cameras[*(camera_system->current_camera_idx)]->SetExposure(value);
		cout << "Setting Current Exposure camera " << *(camera_system->current_camera_idx) <<
			" , Desired " << value << ", Actual " <<
			optitrack_cameras[*(camera_system->current_camera_idx)]->Exposure() << endl;
		Sleep(2);
	}

	void callbackSetCurThreshold(int value, void* data) {
		lpt::Optitrack* camera_system = static_cast<lpt::Optitrack*> (data);
		vector<std::shared_ptr<CameraLibrary::Camera>> optitrack_cameras = camera_system->getOptitrackCameras();
		optitrack_cameras[*(camera_system->current_camera_idx)]->SetThreshold(value);
		cout << "Setting Current Threshold camera " << *(camera_system->current_camera_idx) <<
			" , Desired " << value << ", Actual " <<
			optitrack_cameras[*(camera_system->current_camera_idx)]->Threshold() << endl;
		Sleep(2);
	}

	void callbackSetThreshold(int value, void* data) {
		/*vector<CameraLibrary::Camera*>& optitrack_cameras = *static_cast<vector<CameraLibrary::Camera*>*> (data);*/
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		//value += optitrack_cameras[0]->MinimumThreshold();
		cout << "Setting Threshold Desired = " << value;
		for (int id = 0; id < optitrack_cameras.size(); id++) {
			optitrack_cameras[id]->SetThreshold(value);
			cout << ", " << optitrack_cameras[id]->Threshold();
		}
		cout << endl;
		Sleep(2);
	}

	void callbackSetIRFilter(int state, void* data) {
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		for (int id = 0; id < optitrack_cameras.size(); id++) {
			if (optitrack_cameras[id]->IsFilterSwitchAvailable())
				optitrack_cameras[id]->SetIRFilter(state ? true : false);
		}
		cout << "Setting IR Filter: " << (state ? "on" : "off") << endl;
	}

	void callbackSetAEC(int state, void* data) {
		// Enable or Disable autmatic exposure control AEC
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		cout << "Setting Auto Exposure Control (AEC):" << (state ? "on " : "off ");
		for (int id = 0; id < optitrack_cameras.size(); id++) {
			optitrack_cameras[id]->SetAEC(state ? true : false);
			cout << "," << (state ? "on " : "off ");
		}
		cout << endl;
	}

	void callbackSetAGC(int state, void* data) {
		// Enable or Disable autmatic gain control AGC
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		for (int id = 0; id < optitrack_cameras.size(); id++)
			optitrack_cameras[id]->SetAGC(state ? true : false);
		cout << "Setting Auto Gain Control (AGC): " << (state ? "on" : "off") << endl;
	}

	void callbackSetTextOverlay(int state, void* data) {
		// Enable or Disable text overlay
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		for (int id = 0; id < optitrack_cameras.size(); id++)
			optitrack_cameras[id]->SetTextOverlay(state ? true : false);
		cout << "Setting Text Overlay: " << (state ? "on" : "off") << endl;
	}

	void callbackSetExposure(int value, void* data) {
		// Sets Exposure manually when AEC is not activated
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		//value += optitrack_cameras[0]->MinimumExposureValue();	
		cout << "Setting Exposure Desired = " << value;
		for (int id = 0; id < optitrack_cameras.size(); id++) {
			optitrack_cameras[id]->SetExposure(value);
			cout << " , " << optitrack_cameras[id]->Exposure();
			//<<" (" << optitrack_cameras[id]->DataRate() << "," << optitrack_cameras[id]->PacketSize() << ") ";
		}
		cout << endl;
		Sleep(2);
		//int actual = optitrack_cameras[0]->Exposure();		
		//cout << ", Actual = "<<  actual << "("<< 17.13 * actual << " micro seconds)" << endl; //Only valid for V120:SLIM
	}

	void callbackSetIntensity(int value, void* data) {
		// Sets IR illumination intensity for optitrack_cameras that support this feature
		auto& optitrack_cameras = *static_cast<vector<std::shared_ptr<CameraLibrary::Camera>>*> (data);
		cout << "Setting Intensity: " << "Desired = " << value;

		for (int id = 0; id < optitrack_cameras.size(); id++)
			optitrack_cameras[id]->SetIntensity(value);
		Sleep(2);
		cout << ", Actual = " << optitrack_cameras[0]->Intensity() << endl;
	}

	void callbackSetFrameRate(int value, void* data) {
		lpt::Optitrack* camera_system = static_cast<lpt::Optitrack*> (data);

		if (camera_system) {
			int fps = value;
			//mahdi old framerate model with specified values is commented below
			//switch (value) {
			//	//test for add 240 frame rate		
			//case 3:
			//	fps = 240;
			//	break;
			//	//
			//case 2:
			//	fps = 120;
			//	break;
			//case 1:
			//	fps = 60;
			//	break;
			//case 0:
			//	fps = 30;//mahdi framerate change here
			//	break;
			//default:
			//	fps = 120;
			//	break;
			//}

			// Check if the desired frame rate is smaller than 30
			if (fps < 30) {
				cout << "Desired frame rate " << fps << " is below 30";
				while (30 % fps != 0) {
					fps = fps + 1;
				}
				cout << " Setting to " << fps << endl;
				vector<std::shared_ptr<CameraLibrary::Camera>> optitrack_cameras = camera_system->getOptitrackCameras();
				for (int id = 0; id < optitrack_cameras.size(); id++) { //mahdi tried adding high power mode here but was not supported
					optitrack_cameras[id]->SetFrameRate(30);					
					//mahdi setting grayscale floor no diff
					//int thresholdIntensity = 0; // Example intensity value
					//cout << " now setting grayscale floor for cam#" << id << endl;
					//optitrack_cameras[id]->SetGrayscaleFloor(thresholdIntensity); // Set the grayscale floor
				}
				Sleep(2);
				cout << ", but Actual FPS = " << optitrack_cameras[0]->ActualFrameRate() << endl;
				//camera_system->shared_objects->frame_rate = optitrack_cameras[0]->ActualFrameRate();
				camera_system->shared_objects->frame_rate = fps;
			}
			else {
				//cout << "Setting Frame Rate: Desired = " << fps << endl;
				vector<std::shared_ptr<CameraLibrary::Camera>> optitrack_cameras = camera_system->getOptitrackCameras();
				cout << "Setting fps Desired = " << fps;
				for (int id = 0; id < optitrack_cameras.size(); id++) { //mahdi tried adding high power mode here but was not supported
					optitrack_cameras[id]->SetFrameRate(fps); //mahdi setting grayscale floor no diff
					//int thresholdIntensity = 0; // Example intensity value
					//cout << " now setting grayscale floor for cam#" << id << endl;
					//optitrack_cameras[id]->SetGrayscaleFloor(thresholdIntensity); // Set the grayscale floor
					cout << " , " << optitrack_cameras[id]->ActualFrameRate();
				}
				Sleep(2);
				cout << endl;
				camera_system->shared_objects->frame_rate = optitrack_cameras[0]->ActualFrameRate();
			}
		}
	}

	void callbackSetWritingRawData(int state, void* data) { //mahdi for raw data write
		Optitrack* system = static_cast<Optitrack*> (data);
		system->WriteRawDataRequested = state;
		cout << "Setting write raw image data from camera: " << (state ? "on" : "off") << endl;
	}

#endif /*USE_NP_CAMERASDK*/

	void callbackSetImageViewStatus(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system) {
			system->setImageViewStatus((state != 0));
			cout << "Setting ImageViewStatus: " << (state != 0) << endl;
		}
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Set ImageViewStatus" << endl;

	}

	void callbackSetCompositeView(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system)
			system->setCompositeView((state != 0));
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Set CompositeView" << endl;
	}

	void callbackSetDetectionView(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system)
			system->setDetectionView((state != 0));
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Set DetectionView" << endl;
	}

	void callbackSetReprojectionView(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system)
			system->setReprojectionView((state != 0));
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Set ReprojectionView" << endl;
	}

	void callbackSetTrajectoryView(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system) {
			system->tracker->clear_drawings = true;
			system->setTrajectoryView((state != 0));
		}
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Set TrajectoryView" << endl;
	}

	void callbackFlushFrameQueue(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system) {
			system->frame_queue.clear();
			cout << "Flushed frame queue" << endl;
		}
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot flush queue!!" << endl;
	}

	void callbackFlushProcessedQueue(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system) {
			system->processed_queue.clear();
			cout << "Flushed processed queue" << endl;
		}
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot flush queue!!" << endl;
	}

	void callbackStopCameras(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system)
			system->getCameraSystem()->setCamerasStatus(false);
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Stop!!" << endl;
	}

	void callbackSetXYRFilter(int state, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		if (system) {
			system->setXYRFilter((state != 0));
			cout << "Setting XYR Filter: " << (state != 0) << endl;
		}
		else
			cout << "---INVALID pointer to CameraSystem:  Cannot Set XYRFilter" << endl;
	}

	void callbackSetMinRadius(int Value, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		system->minParticleRadius = (double)system->minParticleRadius_level / 10.0;
		cout << "Setting minParticleRadius to " << system->minParticleRadius << endl;
	}

	void callbackSetMaxRadius(int Value, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		system->maxParticleRadius = (double)system->maxParticleRadius_level / 10.0;
		cout << "Setting maxParticleRadius to " << system->maxParticleRadius << endl;
	}

	void callbackSet_min_x_pos(int Value, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		system->minParticle_x_pos = Value;
		cout << "Setting minParticle_x_pos to " << system->minParticle_x_pos << endl;
	}

	void callbackSet_max_x_pos(int Value, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		system->maxParticle_x_pos = Value;
		cout << "Setting maxParticle_x_pos to " << system->maxParticle_x_pos << endl;
	}

	void callbackSet_min_y_pos(int Value, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		system->minParticle_y_pos = Value;
		cout << "Setting minParticle_y_pos to " << system->minParticle_y_pos << endl;
	}

	void callbackSet_max_y_pos(int Value, void* data) {
		StreamingPipeline* system = static_cast<StreamingPipeline*> (data);
		system->maxParticle_y_pos = Value;
		cout << "Setting maxParticle_y_pos to " << system->maxParticle_y_pos << endl;
	}

	/*****lpt::StreamingPipeline class implementation*****/

	bool StreamingPipeline::initialize() {
		cout << "Initializing streaming pipeline " << endl;
		frame_queue.setCapacity(queue_capacity);
		processed_queue.setCapacity(queue_capacity);
		match_queue.setCapacity(queue_capacity);
		frame3D_queue.setCapacity(queue_capacity);
		monitor_queue.setCapacity(queue_capacity);

		cout << "Concurrent Queue capacities set to " << queue_capacity << endl;
		cout << "--------In dataacquistion.cpp------------" << endl;
		bool cameras_ok = false;

		if (camera_system)
			cameras_ok = camera_system->initializeCameras();
		else
			cout << "No camera system found. Make sure to call StreamingPipeline::attachCameraSystem()" << endl;

		if (cameras_ok) {
			camera_displayed = 0;
			auto& cameras = shared_objects->cameras;
			auto& camera_pairs = shared_objects->camera_pairs;
			auto& output_path = shared_objects->output_path;
			if (!visualizer)
				this->visualizer = std::make_shared < lpt::Visualizer >();
			this->visualizer->setSharedObjects(this->shared_objects);
			this->visualizer->initialize();

			this->calibrator = std::make_shared < lpt::Calibrator >(cameras, camera_pairs, camera_displayed, shared_objects->output_path);
			this->calibrator->setCalibViews(shared_objects->image_type); // mahdi setting half image type for calibration always in mjpeg mode
			this->matcher->initialize();
			this->tracker->setTrajectoryViews(cameras, shared_objects->image_type);
			this->recorder = std::make_shared < lpt::Recorder >(cameras.size(), output_path);
			this->reconstructor = std::make_shared < lpt::Reconstruct3D >(); //std::make_shared < lpt::Reconstruct3DwithSVD > ();
			this->reconstructor->setSharedObjects(this->shared_objects);			
			camera_system->setCalibrator(calibrator); //accessing calibrator inside camera system to access gteframeInterval for virtual file
			//if ( camera_pairs.empty() )       
			//	lpt::generateCameraPairs(this->pt_cameras, this->camera_pairs);            
			camera_system->setCurrentCameraIndex(camera_displayed); // mahdi setting current camera index for separate threhsold and exposure
			initializeControlWindow();			
		}

		return cameras_ok;
	}

	void StreamingPipeline::initializeControlWindow() {
		auto& cameras = shared_objects->cameras;

		if (cameras.empty()) {
			cout << "Could not initialize control window: No cameras found" << endl;
			return;
		}
		cout << "Initializing Control Window with " << cameras.size() << " Cameras" << endl;
		// Set up display window using opencv
		string null;
		camera_displayed = 0;       // index of initial camera to be displayed in opencv window

		image_view_status = true;
		composite_view_status = false;
		detection_view_status = true;
		reprojection_view_status = false;
		trajectory_view_status = false;
		xyr_filter_status = false;

		void* system_void_ptr = static_cast<void*> (this);
		void* recorder_void_ptr = static_cast<void*> (this->recorder.get());

		cv::namedWindow(camera_system->getWindowName(), cv::WINDOW_NORMAL);
		cv::createTrackbar("Camera", camera_system->getWindowName(), &camera_displayed, static_cast<int>(cameras.size() - 1), 0);

		cv::createButton("Record Video Clip", callbackRecordVideo, recorder_void_ptr, cv::QT_PUSH_BUTTON, 0);
		cv::createButton("Take Snapshot", callbackTakeSnapshot, recorder_void_ptr, cv::QT_PUSH_BUTTON, 0);
		cv::createButton("Stop Cameras", callbackStopCameras, system_void_ptr, cv::QT_PUSH_BUTTON, 0);
		cv::createButton("Show ImageView", callbackSetImageViewStatus, system_void_ptr, cv::QT_CHECKBOX, 1);
		cv::createButton("Show Composite", callbackSetCompositeView, system_void_ptr, cv::QT_CHECKBOX, 0);
		cv::createButton("Show Detected", callbackSetDetectionView, system_void_ptr, cv::QT_CHECKBOX, 1); 
		cv::createButton("Reproject 3D", callbackSetReprojectionView, system_void_ptr, cv::QT_CHECKBOX, 0);
		cv::createButton("XYRFilter", callbackSetXYRFilter, system_void_ptr, cv::QT_CHECKBOX, 0);

		cv::createButton("Flush raw queue", callbackFlushFrameQueue, system_void_ptr, cv::QT_PUSH_BUTTON, 0);
		cv::createButton("Flush proc queue", callbackFlushProcessedQueue, system_void_ptr, cv::QT_PUSH_BUTTON, 0);

		cv::createTrackbar("min radius", null, &minParticleRadius_level, 100, callbackSetMinRadius, system_void_ptr);
		cv::createTrackbar("max radius", null, &maxParticleRadius_level, 300, callbackSetMaxRadius, system_void_ptr);
		cv::createTrackbar("min x", null, &minParticle_x_pos, 1280, callbackSet_min_x_pos, system_void_ptr);
		cv::createTrackbar("max x", null, &maxParticle_x_pos, 1280, callbackSet_max_x_pos, system_void_ptr);
		cv::createTrackbar("min y", null, &minParticle_y_pos, 1024, callbackSet_min_y_pos, system_void_ptr);
		cv::createTrackbar("max y", null, &maxParticle_y_pos, 1024, callbackSet_max_y_pos, system_void_ptr);

		this->camera_system->addControls();
		this->calibrator->addControls();
		this->processor->addControls();
		this->detector->addControls();
		this->matcher->addControls();
		this->tracker->addControls();
		this->visualizer->addControls();

		cv::waitKey(50);
		cout << " end of initializeControlWindow" << endl;
	}

	void StreamingPipeline::runControlWindow() {
		//std::thread::id this_id = std::this_thread::get_id();
		//cout << "Thread id " << this_id << "Displaying images (runControlWindow)" << endl;
		auto& cameras = shared_objects->cameras;
		string window_name = camera_system->getWindowName();
		if (cameras.empty()) {
			cout << " mahdi cAMERAS empty in run control window";
			return;
		}
		// Displays images and controls camera parameters
		if (calibrator) {
			cout << "calibrator is good" << endl;
		}
		if (recorder)
			cout << "recorder is good" << endl;

		int last_frame_index = 0;//why 0??
		long int count = 0, count2 = 0;

		while (camera_system->areCamerasRunning()) {

			lpt::Frame3d_Ptr frame3d;

			/*mahdi my chnages to display without processing until ---------------------------
			frame_queue.wait_and_pop(imageproc_frames);
			if (count2 >= shared_objects->frame_rate / 20) {
				count2 = 0;
				stringstream capturedetails;
				if (this->showCompositeView()) {
					for (int c = 0; c < cameras.size(); ++c) {
						capturedetails.str("");
						capturedetails << "Objects = " << imageproc_frames[c].particles.size() << "\t3D Objects = " << frame3d->objects.size() << "\t monitor_queue size = " << monitor_queue.size();
						cv::imshow(cameras[c].name, imageproc_frames[c].image);
						cv::displayStatusBar(cameras[c].name, capturedetails.str(), 1000);
					}
				}
				else {
					capturedetails << "Camera #" << cameras[camera_displayed].id << ": " << cameras[camera_displayed].name <<
						"\n\t2D# = " << imageproc_frames[camera_displayed].particles.size();
					cv::displayStatusBar(window_name, capturedetails.str(), 1000); //mahdi status bar delay in ms was 1000 changed to 20000 for test
					if (this->getImageViewStatus()) {
						if (!imageproc_frames[camera_displayed].image.empty() &&
							imageproc_frames[camera_displayed].image.cols > 0 &&
							imageproc_frames[camera_displayed].image.rows > 0) {
							cv::imshow(window_name, imageproc_frames[camera_displayed].image);
						}
						else {
							cout << " StreamingPipeline::runControlWindow() Error: Invalid or empty image data in camera " << cameras[camera_displayed].name << endl;
						}
					}
				}
				capturedetails.str("");
				capturedetails << "\t frame_queue size = " << frame_queue.size()
					<< "\t processed_queue size = " << processed_queue.size()
					<< "\t match_queue size = " << match_queue.size()
					<< "\t frame3D_queue size = " << frame3D_queue.size()
					<< "\t monitor_queue size = " << monitor_queue.size();
				cv::displayOverlay(window_name, capturedetails.str(), 100); //mahdi disp overlay delay in ms was 100 changed to 20000 for test

				if (count % 60 == 0)//why 60? why wait??
					cv::waitKey(1);
			}
			++count2;
			++count;


			//---------------------------*/
			if (monitor_queue.try_pop(frame3d)) {
				//cerr << " item picked from monitor_queue under runControlWindow() # " << frame3d->camera_frames[0].frame_index << endl;
				lpt::ImageFrameGroup& camera_frames = frame3d->camera_frames;//here each camera_frame is 8 images

				if (recorder->isSnapShotRequested())
					recorder->takeSnapShot(lpt::getImageVector(camera_frames));
				if (recorder->isVideoRecording())
					recorder->addFramesToVideos(lpt::getImageVector(camera_frames));

				if (calibrator->isStereoDataCollecting())
					if (count % calibrator->getFrameInterval() == 0)
						calibrator->findCalibObject(camera_frames);

				if (calibrator->isIntParamDataCollecting())
					if (count % calibrator->getFrameInterval() == 0)
						calibrator->findCalibBoard(camera_frames[camera_displayed]);

				if (calibrator->isSettingGlobalReference()) {
					bool ref_found = calibrator->findGlobalReference(camera_frames);
					if (ref_found)
						calibrator->setGlobalReference(false);
				}

				if (this->showReprojectionView() && !frame3d->objects.empty()) {
					reconstructor->draw(*frame3d);
				}

				if (count2 >= shared_objects->frame_rate / 20) {
					//cerr << " under count2 >= shared_objects->frame_rate / 20)" << endl;
					count2 = 0;
					stringstream capturedetails;
					if (this->showCompositeView()) {
						for (int c = 0; c < cameras.size(); ++c) {
							capturedetails.str("");
							capturedetails << "Objects = " << camera_frames[c].particles.size() << "\t3D Objects = " << frame3d->objects.size() << "\t monitor_queue size = " << monitor_queue.size();
							cv::imshow(cameras[c].name, camera_frames[c].image);
							cv::displayStatusBar(cameras[c].name, capturedetails.str(), 1000);
						}
					}
					else {
						capturedetails << "Camera #" << cameras[camera_displayed].id << ": " << cameras[camera_displayed].name <<
							"\n\t2D# = " << camera_frames[camera_displayed].particles.size() <<
							"\t3D# = " << frame3d->objects.size() << "\ttraj_queue = " << visualizer->getQueueSize() << "\tren_queue = " << visualizer->getRenderQueueSize();

						cv::displayStatusBar(window_name, capturedetails.str(), 1000); //mahdi status bar delay in ms was 1000 changed to 20000 for test
						if (this->getImageViewStatus()) {
							if (!camera_frames[camera_displayed].image.empty() &&
								camera_frames[camera_displayed].image.cols > 0 &&
								camera_frames[camera_displayed].image.rows > 0) {
								cv::imshow(window_name, camera_frames[camera_displayed].image);
							}
							else {
								cout << " StreamingPipeline::runControlWindow() Error: Invalid or empty image data in camera " << cameras[camera_displayed].name << endl;
							}
						}
					}
					capturedetails.str("");
					capturedetails << "\t frame_queue size = " << frame_queue.size()
						<< "\t processed_queue size = " << processed_queue.size()
						<< "\t match_queue size = " << match_queue.size()
						<< "\t frame3D_queue size = " << frame3D_queue.size()
						<< "\t monitor_queue size = " << monitor_queue.size();
					cv::displayOverlay(window_name, capturedetails.str(), 100); //mahdi disp overlay delay in ms was 100 changed to 20000 for test

					if (count % 60 == 0)//why 60? why wait??
						cv::waitKey(1);
				}
				++count2;
				++count;
				//cout << " under control window count = " << count << " and count 2 = " << count2;
			}
			else {
				cv::waitKey(1);
			}
			// Check for frame queue overload:  this depends on the type of camera since frameID is defined differently
			//if( monitor_queue.front()[camera_displayed].frame_index - last_frame_index != 1) {
			//	stringstream overloaddisplay;
			//	overloaddisplay << "FrameQueue Overload: Queue size = " << monitor_queue.size();
			//	cv::displayOverlay(window_name, overloaddisplay.str(), 100);
			//}
			//last_frame_index = monitor_queue.front()[camera_displayed].frame_index;
		}
		cout << "Control Window thread done" << endl;
		cv::destroyAllWindows();
	}

	void StreamingPipeline::aquireImageData() {
		//std::thread::id this_id = std::this_thread::get_id();
		//std::cout << " mahdi imagegrabber_thread aquireImageData ID: " << this_id << std::endl;
		auto& cameras = shared_objects->cameras;
		cout << "------------Starting " << cameras.size() << " cameras-------------" << endl;
		this->setFrameRate(camera_system->getFrameRate());

		boost::posix_time::microseconds sleeptime(100);

		//mahdi changes for writing to file below
		//ofstream outfile("frame_data.txt");
		//if (!outfile.is_open()) {
		//	std::cout << "Unable to open file for writing." << std::endl;
		//	return;
		//}
		//end mahdi
		while (camera_system->areCamerasRunning()) {
			//auto optitrack = std::dynamic_pointer_cast<Optitrack>(camera_system);
			lpt::ImageFrameGroup frame_group(cameras.size());
			bool good_frames = camera_system->grabFrameGroup(frame_group);
			//mahdi lines below to write time and frame id in a txt file
			//if (camera_system->WriteRawDataRequested) {
			//	if (good_frames) {
			//		//auto now = std::chrono::system_clock::now();
			//		//auto now_c = std::chrono::system_clock::to_time_t(now);
			//		//std::tm now_tm = *std::localtime(&now_c);
			//		outfile << ",frame_i=" << (frame_group[0].frame_index) << std::endl;
			//		//outfile << "at: " << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << ",frame_i=" << (frame_group[0].frame_index) << std::endl;
			//	}
			//}
			//end mahdi
			if (good_frames && !frame_queue.full()) {
				//cerr << " item pushed into frame_queue # " << frame_group[0].frame_index << endl;
				frame_queue.push(std::move(frame_group));
				//system("pause");
			}
			else {
				//boost::this_thread::sleep(sleeptime); //mahdi This sleep operation helps manage the rate of frame acquisition and processing, preventing the queue from overflowing and ensuring that the system can handle the data rate.
				boost::this_thread::sleep_for(boost::chrono::microseconds(100)); //mahdi This sleep operation helps manage the rate of frame acquisition and processing, preventing the queue from overflowing and ensuring that the system can handle the data rate.
			}
		}

		//outfile.close();//mahdi
	}

	void StreamingPipeline::processImages(int index) {
		//boost::thread::id this_id = boost::this_thread::get_id();
		//std::cout << " mahdi imageproc_workers_thread_group processImages ID: " << this_id << std::endl;

		auto& cameras = shared_objects->cameras;
		auto& image_type = shared_objects->image_type;

		cout << "Thread #" << index << " processing images" << endl;
		//	boost::chrono::system_clock::time_point start, stop;
		cv::Mat map1, map2;
		cv::Mat temp_image;
		//cv::Mat original_image;
		cv::Mat mask_image;
		cv::Mat cam_mat = cameras[index].getCameraMatrix();
		cv::Mat dist_coeffs = cameras[index].getDistCoeffs();
		cv::Size image_size = image_type.size();
		//cv::initUndistortRectifyMap(cam_mat, dist_coeffs, cv::Mat(), cv::Mat(), image_size, CV_32FC1, map1, map2);   

		cv::Mat R = cv::Mat(3, 3, CV_64F, shared_objects->cameras[index].R);
		cv::Mat t_vec = cv::Mat(3, 1, CV_64F, shared_objects->cameras[index].T);
		cv::Mat r_vec = cv::Mat::zeros(3, 1, CV_64F);
		cv::Rodrigues(R, r_vec);

		bool first = true;
		int counter = 0;
		while (camera_system->areCamerasRunning()) {
			//vector<cv::Point3d> object_points;  // vector to store the reprojection 3d points commented by mahdi
			//cout << " mahdi this is under void StreamingPipeline::processImages in data acq " << endl;
			boost::unique_lock<boost::mutex> lock(imageproc_mutex, boost::try_to_lock);
			imageproc_barrier->wait();
			if (lock.owns_lock()) {
				//stop = boost::chrono::system_clock::now();
				/*if ( ! first) {
					this->imageproc_time += stop - start;
				} else
					first = false;*/

				if (!processed_queue.full()) {
					//cerr << " item pushed into processed_queue under processImages # " << imageproc_frames[index].frame_index << endl;
					processed_queue.push(std::move(imageproc_frames));
					//cout << " mahdi from imageproc_frames to processed_queue by thread# " << index << endl;
					//mahdi: output varied 2, 4 but remained the same during one program execution
				}
				frame_queue.wait_and_pop(imageproc_frames);
				//cerr << " item picked from frame_q under processImages # " << imageproc_frames[index].frame_index << endl;
				//start = boost::chrono::system_clock::now();
			}

			imageproc_barrier->wait();

			//start = boost::chrono::system_clock::now();

			//if (this->camera_system->getCurParCol(index) == false) {
			//	detector->detectFeatures(imageproc_frames[index].image, mask_image, imageproc_frames[index].particles, imageproc_frames[index].contours, index);
			if (this->camera_system->getParticleCollectionFromCamera() == false) { // This line is for uniform camera system
				temp_image = imageproc_frames[index].image.clone();
				//processor->processImage(temp_image);
				//detector->detectFeatures(temp_image, imageproc_frames[index].image, imageproc_frames[index].particles, imageproc_frames[index].contours, index);
			}

			//imageproc_barrier->wait(); // newlly added
			if (this->getXYRFilter()) {
				lpt::filterParticles(imageproc_frames[index],
					minParticleRadius, maxParticleRadius,
					minParticle_x_pos, maxParticle_x_pos, minParticle_y_pos, maxParticle_y_pos);
			}
			if (this->showDetectionView()) {
				detector->drawResult(imageproc_frames[index]);   // TODO Modify this later to see if the particle image is right //mahdi: replaced with rectangle, correct
				//detector->drawContours( imageproc_frames[index].image, imageproc_frames[index].contours );
			}

			// comment for self-calibration
			lpt::undistortPoints(cameras[index], imageproc_frames[index]);

			boost::this_thread::interruption_point();
		}

		cout << index << " Image Processor thread done:" << endl; //<< this->imageproc_time << endl;
	}

	void StreamingPipeline::solveCorrespondence() {
		//std::thread::id this_id = std::this_thread::get_id();
		//std::cout << " mahdi matcher_thread solveCorrespondence ID: " << this_id << std::endl;
		cout << "Correspondence Thread started" << endl;

		// load the pairs map
		matcher->initiate_pairmap();

		//	boost::chrono::system_clock::time_point start, stop;
		//	start = boost::chrono::system_clock::now();
		matcher->run(&processed_queue, &match_queue, 1, 1);

		//	stop = boost::chrono::system_clock::now();
		//	this->correspond_time = stop - start;
		cout << "Matching thread done: " << endl; //this->correspond_time << endl;
	}

	void StreamingPipeline::reconstuct3DObjects() {
		//std::thread::id this_id = std::this_thread::get_id();
		//std::cout << " mahdi reconstructor_thread reconstuct3DObjects ID: " << this_id << std::endl;
		cout << "Reconstruct 3D Objects Thread started" << endl;
		//	boost::chrono::system_clock::time_point start, stop;
		while (camera_system->areCamerasRunning()) {
			std::pair<lpt::ImageFrameGroup, vector<lpt::Match::Ptr> > newpair;
			match_queue.wait_and_pop(newpair);
			//cerr << " item picked from matchq under reconstuct3DObjects() #" << newpair.first[0].frame_index << endl;
			//start = boost::chrono::system_clock::now();
			lpt::ImageFrameGroup& frame_group = newpair.first;
			vector<lpt::Match::Ptr>& matches = newpair.second;

			if (this->visualizer->getTakeImageMeasurement()) {
				cout << " mahdi this is inside get take measutment" << endl;
				this->visualizer->accumulateCentroidDetectionUncertainty(matches);
			}

			lpt::Frame3d_Ptr newframe3d = lpt::Frame3d::create(frame_group, frame_group[0].frame_index);
			reconstructor->reconstruct3DFrame(matches, *newframe3d);

			//if (this->showReprojectionView() && !newframe3d->objects.empty()) {
			//	reconstructor->calculateReprojectionError(matches, *newframe3d);
			//}

			if (!frame3D_queue.full()) {
				//cerr << " item pushed to frame3dQ under reconstuct3DObjects()# " << newframe3d->camera_frames[0].frame_index << endl;
				frame3D_queue.push(std::move(newframe3d));
			}
			//stop = boost::chrono::system_clock::now();
			//this->recon_time += stop - start;
			boost::this_thread::interruption_point();
		}
		//stop = boost::chrono::system_clock::now();
		cout << "3D reconstruction thread done: " << endl;//this->recon_time <<  endl;
	}

	void StreamingPipeline::trackObjects() {
		//std::thread::id this_id = std::this_thread::get_id();
		//std::cout << " mahdi tracker_thread trackObjects ID: " << this_id << std::endl;
		cout << "Track Objects Thread started" << endl;
		//boost::posix_time::microseconds sleeptime(500);
		//	boost::chrono::system_clock::time_point start, stop;
		int count = 0;
		int last_frame_index = 0;
		lpt::Frame3d_Ptr frame1, frame2;
		lpt::Frame3d_Ptr reprojectFrame;

		while (camera_system->areCamerasRunning()) {
			if (!frame1) {
				// wait barrier here
				frame3D_queue.wait_and_pop(frame1);
				//mahdi: this never happens as there is the empty frame1 in frame3dQ with the id=-1
				//cerr << " item picked from frame3D_queue as frame 1 under trackObjects # " << frame1->frame_index << endl;
			}
			// wait barrier here
			frame3D_queue.wait_and_pop(frame2);
			//cerr << " item picked from frame3D_queue as frame 2 under trackObjects # " << frame2->frame_index << " and frame 1# " << frame1->frame_index << endl;
			// the 3d frame to be pushed back to reprojection queue
			reprojectFrame = lpt::Frame3d::create(frame2->frame_index);


			//// begin by mahdi
			//// writing frame2 points to tecplot output
			//std::array<double, 3> GridOrigin = { {-600, 0, -250} };
			//std::array<int, 3> gridDim = { {1500, 1500, 500} };
			//std::array<int, 3> gridCellCounts = { {70, 70, 1} };

			//// Calculate the cell sizes
			//double dx = gridDim[0] / static_cast<double>(gridCellCounts[0]);
			//double dy = gridDim[1] / static_cast<double>(gridCellCounts[1]);
			//double dz = gridDim[2] / static_cast<double>(gridCellCounts[2]);


			//string filename = "frame3d\\frame_" + to_string(frame2->frame_index) + ".plt";
			//ofstream outfile(filename);
			//if (!outfile.is_open()) cout << "failed to open output frame write plt " << endl;
			//outfile << "TITLE = \"3D Points\"" << std::endl;
			//outfile << "VARIABLES = \"X\" \"Y\" \"Z\"\n";

			//outfile << "ZONE T=\"Points\"" << std::endl;
			//for (auto point : frame2->objects) {
			//	outfile << point->X[0] << " " << point->X[1] << " " << point->X[2] << std::endl;
			//}
			////// Writing the grid whole of it
			////outfile << "ZONE T=\"Grid\", I=" << gridCellCounts[0] << ", J=" << gridCellCounts[1] << ", K=" << gridCellCounts[2] << ", DATAPACKING=POINT\n";
			////for (int k = 0; k < gridCellCounts[2]; k++) {
			////	for (int j = 0; j < gridCellCounts[1]; j++) {
			////		for (int i = 0; i < gridCellCounts[0]; i++) {
			////			double x = GridOrigin[0] + i * dx;
			////			double y = GridOrigin[1] + j * dy;
			////			double z = GridOrigin[2] + k * dz;
			////			outfile << x << " " << y << " " << z;
			////			if (k < gridCellCounts[2] - 1 || j < gridCellCounts[1] - 1 || i < gridCellCounts[0] - 1)
			////				outfile << "\n";
			////		}
			////	}
			////}


			//outfile << "ZONE T=\"Grid\", I=2, J=2, K=2, DATAPACKING=POINT\n";

			//// Corners of the cuboid
			//outfile << GridOrigin[0] << " " << GridOrigin[1] << " " << GridOrigin[2] << "\n";   // Corner 1
			//outfile << GridOrigin[0] + dx * (gridCellCounts[0] - 1) << " " << GridOrigin[1] << " " << GridOrigin[2] << "\n";  // Corner 2
			//outfile << GridOrigin[0] << " " << GridOrigin[1] + dy * (gridCellCounts[1] - 1) << " " << GridOrigin[2] << "\n";  // Corner 3
			//outfile << GridOrigin[0] + dx * (gridCellCounts[0] - 1) << " " << GridOrigin[1] + dy * (gridCellCounts[1] - 1) << " " << GridOrigin[2] << "\n";  // Corner 4
			//// Repeating above for top layer since K=2, assuming the cuboid extends upwards in Z
			//outfile << GridOrigin[0] << " " << GridOrigin[1] << " " << (GridOrigin[2] + dz) << "\n";   // Corner 5
			//outfile << GridOrigin[0] + dx * (gridCellCounts[0] - 1) << " " << GridOrigin[1] << " " << (GridOrigin[2] + dz) << "\n";  // Corner 6
			//outfile << GridOrigin[0] << " " << GridOrigin[1] + dy * (gridCellCounts[1] - 1) << " " << (GridOrigin[2] + dz) << "\n";  // Corner 7
			//outfile << GridOrigin[0] + dx * (gridCellCounts[0] - 1) << " " << GridOrigin[1] + dy * (gridCellCounts[1] - 1) << " " << (GridOrigin[2] + dz) << "\n";  // Corner 8



			//////only corner points
			////	// Lower left corner
			////outfile << GridOrigin[0] << " " << GridOrigin[1] << " " << GridOrigin[2] << "\n";
			////// Lower right corner
			////outfile << (GridOrigin[0] + dx * (gridCellCounts[0] - 1)) << " " << GridOrigin[1] << " " << GridOrigin[2] << "\n";
			////// Upper left corner
			////outfile << GridOrigin[0] << " " << (GridOrigin[1] + dy * (gridCellCounts[1] - 1)) << " " << GridOrigin[2] << "\n";
			////// Upper right corner
			////outfile << (GridOrigin[0] + dx * (gridCellCounts[0] - 1)) << " " << (GridOrigin[1] + dy * (gridCellCounts[1] - 1)) << " " << GridOrigin[2] << "\n";
			////// Two more points (duplicates) for fulfilling the six points requirement
			////// Middle left corner
			////outfile << GridOrigin[0] << " " << (GridOrigin[1] + dy * (gridCellCounts[1] / 2)) << " " << GridOrigin[2] << "\n";
			////// Middle right corner
			////outfile << (GridOrigin[0] + dx * (gridCellCounts[0] - 1)) << " " << (GridOrigin[1] + dy * (gridCellCounts[1] / 2)) << " " << GridOrigin[2] << "\n";
			//outfile.close();
			//std::cout << "Data written to " << filename << std::endl;
			//// end by mahdi
			//start = boost::chrono::system_clock::now();		
			if (last_frame_index != frame1->frame_index) {
				cout << "!!!!!!!!!!Frame skipped!!!!!!!!!!!!! " << frame1->frame_index << endl;
				last_frame_index = frame2->frame_index;
				frame1 = frame2;
				continue;
			}

			last_frame_index = frame2->frame_index;
			//cerr << "last_frame_index = frame2->frame_index=" << last_frame_index << endl;
			tracker->trackFrames(*frame1, *frame2); // once the tracking is done, project back to 2d space and create a mask for next frame index corner detection

			if (visualizer->getVisualizationStatus())
				visualizer->addTrajectoriesToQueue(tracker->getActiveTrajectories());

			if (!monitor_queue.full()) {
				//cerr << " item pushed into monitor_queue under trackObjects # " << frame1->frame_index << endl;
				monitor_queue.push(std::move(frame1));
			}
			frame1 = frame2;
			++count;
			//stop = boost::chrono::system_clock::now();	
			//this->tracking_time += stop - start;

			boost::this_thread::interruption_point();
		}
		//tracker->finalizeTrajectoryList();
		cout << "tracking thread done: " << endl;// this->tracking_time << endl;
	}

	void StreamingPipeline::run() { // mahdi this is the main loop run
		//DWORD mainThreadId = GetCurrentThreadId(); // Get the thread ID of the calling thread
		//std::cout << " mahdi StreamingPipeline::run() MAIN Thread ID: " << mainThreadId << std::endl;
		auto& cameras = shared_objects->cameras;

		// initiate the vectors for gfft
		/*if (detector->isGfftDetector()) {
			lpt::Detector* tempDetectorPtr = &(*detector);
			lpt::GoodFeaturesToTrackDetector* gfftDetector = dynamic_cast<lpt::GoodFeaturesToTrackDetector*>(tempDetectorPtr);
			gfftDetector->initCudaVecs(cameras.size(), shared_objects->image_type, shared_objects->half_image_type);
		}*/
		imagegrabber_thread = boost::thread(&StreamingPipeline::aquireImageData, this);
		cout << "-----------thread for acquire ImageData is created!" << endl;
		imageproc_barrier = std::make_shared<boost::barrier>(cameras.size());

		lpt::ImageFrame init_frame;
		init_frame.image = shared_objects->image_type;

		imageproc_frames.resize(cameras.size(), init_frame);
		for (int c = 0; c < cameras.size(); ++c)
			imageproc_workers_thread_group.create_thread(boost::bind(&StreamingPipeline::processImages, this, c));

		
		matcher_thread = boost::thread(&StreamingPipeline::solveCorrespondence, this);
		reconstructor_thread = boost::thread(&StreamingPipeline::reconstuct3DObjects, this);
		tracker_thread = boost::thread(&StreamingPipeline::trackObjects, this);
		//visualizer->setCameras(cameras);
		//visualizer_thread = boost::thread(&StreamingPipeline::runVisualizer, this);
		
#ifdef WIN32
		// Priority settings available: THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_ABOVE_NORMAL,  
		//	  THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_LOWEST

		SetThreadPriority(imagegrabber_thread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

		HANDLE mainthread = GetCurrentThread();
		SetThreadPriority(mainthread, THREAD_PRIORITY_TIME_CRITICAL);
		SetThreadPriority(matcher_thread.native_handle(), THREAD_PRIORITY_HIGHEST);
		SetThreadPriority(reconstructor_thread.native_handle(), THREAD_PRIORITY_HIGHEST);
		SetThreadPriority(tracker_thread.native_handle(), THREAD_PRIORITY_HIGHEST);

#else 
		cout << "need to set thread priorities for Linux systems" << endl;
#endif

		boost::this_thread::sleep_for(boost::chrono::seconds(3));
		//boost::this_thread::sleep(boost::posix_time::seconds(3));
		this->runControlWindow();
		// All threads running...stop called when cameras are stopped
		this->stop();
	}

	void StreamingPipeline::stop() {

		imagegrabber_thread.join();

		imageproc_workers_thread_group.interrupt_all();
		imageproc_workers_thread_group.join_all();

		matcher_thread.interrupt();
		matcher_thread.join();
		matcher->stop();

		reconstructor_thread.interrupt();
		reconstructor_thread.join();

		tracker_thread.interrupt();
		tracker_thread.join();
		this->camera_system->shutdown();

		visualizer_thread.join();
	}

	void StreamingPipeline::load_Rotation_Matrix_virtual() {
		ifstream S_in, P_in;
		string S = this->shared_objects->input_path + "S_virtual.txt";
		string P = this->shared_objects->input_path + "P_virtual.txt";
		S_in.open(S);
		P_in.open(P);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				S_in >> this->shared_objects->S[i][j];
			}
			P_in >> this->shared_objects->P[i];
		}
		this->shared_objects->isRotation_Correction = false;
	}

	void StreamingPipeline::load_Rotation_Matrix()
	{
		ifstream S_in, P_in;
		string S = this->shared_objects->input_path + "S.txt";
		string P = this->shared_objects->input_path + "P.txt";
		S_in.open(S);
		P_in.open(P);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				S_in >> this->shared_objects->S[i][j];
			}
			P_in >> this->shared_objects->P[i];
		}
		this->shared_objects->isRotation_Correction = true;
	}

} /*NAMESPACE PT*/