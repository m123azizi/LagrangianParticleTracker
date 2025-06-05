#include <windows.h> //mahdi
#include "correspond.hpp"

namespace lpt {

	using namespace std;


	Correspondence::Correspondence()
		: initial_max_particles_per_image(2048), map_storage_size(500) //mahdi changing max particles per image
		//: initial_max_particles_per_image(500), map_storage_size(300)
	{
		positiveData.open("../../../data/output/positive_data.txt");
		negativeData.open("../../../data/output/negative_data.txt");
		imageData.open("../../../data/output/image_data.txt");

		//lr_data_path = "C:/pytorchTest/LR.txt";

		// test torch module
		//torch_module = torch::jit::load("C:/pytorchTest/traced_model.pt");
		//torch_module.to(at::kCUDA);
		//torch_module.eval();

		//// test the torch model to see if it is the same as pytorch
		//vector<torch::jit::IValue> inputs;
		//auto options = torch::TensorOptions().dtype(at::kFloat);
		//at::Tensor testTensor = torch::ones({ 1, 10 }, options).to(at::kCUDA);

		//inputs.push_back(testTensor);

		//torch::NoGradGuard no_grad;
		//at::Tensor output = torch_module.forward(inputs).toTensor();

		//cout << "Test result of torch model is: " << output << endl;

		//cout << "----------Succeed to load torch module-------------" << endl;

		//read_pca("C:/pytorchTest/data/pca_svm/pca_parameters.txt");
		//read_train_mean_std("C:/pytorchTest/data/pca_svm/mean_x_train.txt", "C:/pytorchTest/data/pca_svm/std_x_train.txt");

		//read_lr(lr_data_path);

		//// test of scaling
		//auto options = torch::TensorOptions().dtype(at::kFloat);
		//at::Tensor test_ones = torch::ones({ 1, 30 }, options);

		//test_ones.to(at::kCUDA);
		//test_ones = (test_ones - mean_tensor) / std_tensor;

		//cout << "Test of scaling before pca" << endl;
		//cout << test_ones << endl;


		//test_ones = at::matmul(test_ones, pca_tensor).to(at::kCUDA);

		//cout << "Test of scaling after pca" << endl;
		//cout << test_ones << endl;

		//grayscale_camera_ids.push_back(1);
		//grayscale_camera_ids.push_back(5);

		num_grayscale_cameras = grayscale_camera_ids.size();
		//cout << "mahdi under num grayscale cam ids = " << num_grayscale_cameras << endl; //mahdi
		for (int ii = 0; ii < grayscale_camera_ids.size(); ++ii)
			cout << "mahdi under grayscale cam ids = " << grayscale_camera_ids[ii] << endl; //mahdi
	}

	Correspondence::~Correspondence() {
		//positiveData.close();
		//negativeData.close();
	}

	////void Correspondence::read_pca(string path) {
	//	ifstream fileStream(path);
	//
	//	if (!fileStream.is_open()) {
	//		cout << "Fail to read pca parameters file " << path << endl;
	//		exit(1);
	//	}
	//
	//	string line;
	//
	//	while (getline(fileStream, line, '\n')) {
	//		stringstream ss(line);
	//		vector<float> numbers;
	//		string number;
	//
	//		float temp_number = 0.0;
	//
	//		while (getline(ss, number, ' ')) {
	//			temp_number = stof(number);
	//			numbers.push_back(temp_number);
	//		}
	//
	//		pca_parameters.push_back(numbers);
	//	}
	//
	//	pca_rows = pca_parameters.size();
	//	pca_cols = pca_parameters[0].size();
	//
	//	auto options = torch::TensorOptions().dtype(at::kFloat);
	//
	//	pca_tensor = torch::ones({ pca_rows, pca_cols }, options);
	//
	//
	//	for (int i = 0; i < pca_rows; i++)
	//		pca_tensor.slice(0, i, i + 1) = torch::from_blob(pca_parameters[i].data(), { pca_cols }, options);
	//
	//
	//	pca_tensor.to(at::kCUDA);
	//
	//	//cout << pca_tensor << endl;
	//
	//	cout << "Succeed to read in pca parameters" << endl;
	//
	//
	//}

	//void Correspondence::read_lr(string path) {
	//	ifstream fileStream_mean(path);
	//
	//	if (!fileStream_mean.is_open()) {
	//		cout << "Fail to read x_train mean file " << path << endl;
	//		exit(1);
	//	}
	//
	//	string line;
	//
	//	while (getline(fileStream_mean, line, '\n')) {
	//		lr_data.push_back(stof(line));
	//	}
	//
	//	cout << lr_data << endl;
	//
	//	cout << "Succeed to read lr data" << endl;
	//}

	//void Correspondence::read_train_mean_std(string mean_path, string std_path) {
	//	ifstream fileStream_mean(mean_path);
	//
	//	if (!fileStream_mean.is_open()) {
	//		cout << "Fail to read x_train mean file " << mean_path << endl;
	//		exit(1);
	//	}
	//
	//	string line;
	//
	//	while (getline(fileStream_mean, line, '\n')) {
	//		x_train_mean.push_back(stof(line));
	//	}
	//
	//	ifstream fileStream_std(std_path);
	//	if (!fileStream_std.is_open()) {
	//		cout << "Fail to read x_train std file " << std_path << endl;
	//		exit(1);
	//	}
	//
	//	while (getline(fileStream_std, line, '\n')) {
	//		x_train_std.push_back(stof(line));
	//	}
	//
	//	int mean_std_size = (x_train_mean.size() == x_train_std.size() ? x_train_mean.size() : -1);
	//
	//	if (mean_std_size > 0) {
	//		auto options = torch::TensorOptions().dtype(at::kFloat);
	//
	//		mean_tensor = torch::ones({ 1, mean_std_size }, options);
	//		std_tensor = torch::ones({ 1, mean_std_size }, options);
	//
	//		mean_tensor = torch::from_blob(x_train_mean.data(), { 1, mean_std_size }, options);
	//		std_tensor = torch::from_blob(x_train_std.data(), { 1, mean_std_size }, options);
	//
	//		mean_tensor.to(at::kCUDA);
	//		std_tensor.to(at::kCUDA);
	//
	//	}
	//
	//	//cout << mean_tensor << endl;
	//	//cout << std_tensor << endl;
	//
	//	cout << "Succeed to read mean and std of train data" << endl;
	//}

	void Correspondence::findMatches(const ImageFrameGroup& frame_group, vector<Match::Ptr>& matches)
	{
		cout << "mahdi this is under corrs findMatches -----------------------------------" << endl; //mahdi: this is never called
		resetMatchMap(current_matchmap);
		findEpipolarMatches(frame_group, current_matchmap);
		findUniqueMatches(frame_group, current_matchmap, matches);
		//find3WayMatches(frame_group, current_matchmap, matches);
	}

	void Correspondence::findMatches_sequence(const ImageFrameGroup& frame_group, vector<Match::Ptr>& matches)
	{
		cout << "mahdi this is under findmatches_sequence ---------------------------------" << endl; //mahdi: never called
		resetMatchMap(current_matchmap);
		findEpipolarMatches(frame_group, current_matchmap);
		//findUniqueMatches(frame_group, current_matchmap, matches);
		//find3WayMatches_wo_clear(frame_group, current_matchmap, matches);
		//find2WayMathces_wo_clear(frame_group, current_matchmap, matches);
		//find2WayMathces(frame_group, current_matchmap, matches);
	}


	void  Correspondence::run(
		lpt::concurrent_queue<lpt::ImageFrameGroup>* in_queue,
		lpt::concurrent_queue<std::pair<lpt::ImageFrameGroup, vector<lpt::Match::Ptr> > >* out_queue,
		int number_epipolarmatching_threads,
		int number_uniquematching_threads)
	{
		cout << " mahdi Running correspondence with " << number_epipolarmatching_threads << " epipolarmatching threads, and " << number_uniquematching_threads << " unique match threads " << endl;
		//std::cout << " --------------------------------- mahdi under correpondence main run " << endl;
		//for (int i = 0; i < number_2way_threads; ++i)
		//	epipolar_match_threads.create_thread( boost::bind( &Correspondence::runEpipolarMatching, this, 1, in_queue ) );
		//for (int i = 0; i < number_4way_threads; ++i)
		//	unique_match_threads.create_thread( boost::bind( &Correspondence::runUniqueMatching, this, 1) );

		boost::thread epipolar_matcher_th = boost::thread(boost::bind(&Correspondence::runEpipolarMatching, this, 1, in_queue));

		runUniqueMatching(1, out_queue);

		//while (true) {
		//	// order output from 4way threads and send to out_queue
		//	lpt::ImageFrameGroup frame_group;
		//	vector<lpt::Match::Ptr> matches;
		//	auto pair = std::make_pair(frame_group, matches);
		//	final_match_queue.wait_and_pop(pair);
		//	// wait_pop from match_queue containing frame_group, matches pairs
		//	out_queue->push( pair );
		//	boost::this_thread::interruption_point();
		//}
	}

	void Correspondence::stop()
	{
		//epipolar_match_threads.interrupt_all(); //mahdi these thread groups do not work and commented, 1 may 2021
		//epipolar_match_threads.join_all();//mahdi these thread groups do not work and commented, 1 may 2021

		//unique_match_threads.interrupt_all();//mahdi these thread groups do not work and commented, 1 may 2021
		//unique_match_threads.join_all();//mahdi these thread groups do not work and commented, 1 may 2021
	}

	void Correspondence::runEpipolarMatching(int thread_id, lpt::concurrent_queue<lpt::ImageFrameGroup>* in_queue)
	{
		std::thread::id this_id = std::this_thread::get_id();
		//std::cout << " mahdi runEpipolarMatching ID: " << this_id << std::endl;
		this->initializeEpipolarMatchThread(thread_id);
		while (true) {
			//cout << "mahdi this is under runEpipolarMatching while loop in cpp" << endl;
			lpt::ImageFrameGroup frame_group;
			//Sleep(100);
			in_queue->wait_and_pop(frame_group);
			//cerr << " item picked from processed_queue (inQ) under runEpipolarMatching # " << frame_group[0].frame_index << endl;
			vector<pair<lpt::ImageFrameGroup, lpt::MatchMap>>::iterator iter;

			this->empty_maps_queue.wait_and_pop(iter);
			//cerr << " item picked from EMPTYMAPSq (for current_map) under runEpipolarMatching # " << iter->first[0].frame_index << endl;
			lpt::MatchMap& current_map = iter->second;
			resetMatchMap(current_map);
			findEpipolarMatches(frame_group, current_map);
			//cout << " mahdi now calling print matchmap " << endl;
			//printMatchMap(frame_group, current_map, "matchmap_mahdi.txt");
			iter->first = frame_group;
			//cerr << " item pushed into full_maps_queue under runEpipolarMatching # " << iter->first[0].frame_index << endl;
			this->full_maps_queue.push(iter);

			boost::this_thread::interruption_point();
		}
	}

	void Correspondence::runUniqueMatching(int thread_id, lpt::concurrent_queue<std::pair<lpt::ImageFrameGroup, vector<lpt::Match::Ptr> > >* out_queue)
	{
		std::thread::id this_id = std::this_thread::get_id();
		//std::cout << " mahdi runUniqueMatchingThread ID: " << this_id << std::endl;
		while (true) {

			vector<lpt::Match::Ptr> matches;
			vector<pair<lpt::ImageFrameGroup, lpt::MatchMap>>::iterator iter;
			this->full_maps_queue.wait_and_pop(iter);
			//cerr << " item picked from full_maps_queue under runUniqueMatching # " << iter->first[0].frame_index << endl;
			lpt::MatchMap& current_map = iter->second;
			lpt::ImageFrameGroup& group = iter->first;
			//findUniqueMatches(group, current_map, matches);  // at least four cameras

			find3WayMatches_mahdi(group, current_map, matches);  // at least three cameras // mahdi: the initial
			//find2WayMathces(group, current_map, matches);

			//find2WayMathces_wo_clear(group, current_map, matches);

			//find3WayMatches(group, current_map, matches);//07——11——2019 edited, was above before
			// push frame_group and matches to internal match_queue for reordering and pushing to out_queue

			//this->final_match_queue.push(std::make_pair(group, matches));
			//cerr << " item pushed into matchQ(outQ) under runUniqueMatching # " << iter->first[0].frame_index << endl;
			out_queue->push(std::make_pair(group, matches));
			this->empty_maps_queue.push(iter);
			boost::this_thread::interruption_point();
		}
	}

	double Correspondence::calculateEpipolarResidual(double line[3], const lpt::ParticleImage& point) const
	{
		return fabs(point.x * line[0] + point.y * line[1] + line[2]);
	}

	void Correspondence::calculateEpiline(const lpt::ParticleImage& point, const double F[3][3], double line[3]) const
	{
		//cout << "mahdi this is under calculate epiline in correspond cpp" << endl; //mahdi
		double x = point.x;
		double y = point.y;
		double z = 1.0;

		line[0] = F[0][0] * x + F[1][0] * y + F[2][0] * z;
		line[1] = F[0][1] * x + F[1][1] * y + F[2][1] * z;
		line[2] = F[0][2] * x + F[1][2] * y + F[2][2] * z;

		double factor = line[0] * line[0] + line[1] * line[1];
		factor = factor ? 1. / sqrt(factor) : 1.;
		line[0] *= factor;
		line[1] *= factor;
		line[2] *= factor;
	}

	void Correspondence::testMatches(const ImageFrameGroup& cameragroup, const vector<Match::Ptr>& matches) const
	{
		size_t N_correct = 0;
		size_t N_matched = 0;
		size_t N_total = 100000000;
		for (int c = 0; c < cameragroup.size(); c++) {
			size_t temp = cameragroup[c].particles.size();
			cout << "mahdi under test macthes, we have size_t temp smaller than n_total = " << temp << "and c of loop for camera group = " << c << endl;
			if (temp < N_total)
				N_total = temp;
		}

		for (int m = 0; m < matches.size(); m++) {
			if (matches[m]) {
				N_matched++;
				auto& particles = matches[m]->particles;
				bool matched = true;
				for (int j = 0; j < particles.size() - 1; j++) {
					matched &= particles[j].first->id == particles[j + 1].first->id ? true : false;
				}
				if (matched)
					N_correct++;
			}
		}
		cout << "\t Correct Ratio: " << N_correct << " / " << N_matched << " = " <<
			setprecision(2) << (N_matched > 0 ? (double)N_correct / N_matched : 0.0) << endl;
		cout << "\t Cover Ratio: " << N_correct << " / " << N_total << " = " <<
			setprecision(2) << (N_total > 0 ? (double)N_correct / N_total : 0.0) << endl;

	}

	void Correspondence::printMatchMap(const lpt::ImageFrameGroup& frame_group, string output_file_name) const
	{
		//cout << "printing matchmap..." << endl;
		ofstream fout(output_file_name.c_str());
		int p_id = 0;
		for (int i = 0; i < frame_group.size(); ++i) {
			fout << "------------------------------------------" << endl;
			for (int p = 0; p < frame_group[i].particles.size(); ++p, ++p_id) {
				fout << "Cam " << i << ", " << p << ": " << endl;
				for (int c = 0; c < frame_group.size(); ++c) {
					fout << "\tC" << c << " -- ";
					for (int d = 0; d < NUM_MATCHES; ++d) {
						int index = current_matchmap[p_id][c][d];
						if (index >= 0)
							fout << "[" << index << ", " << frame_group[c].particles[index]->id << "]  ";
						else
							fout << index << "  ";
					}
					fout << endl;
				}
			}
		}
		fout.close();
	}
	// similar printMatchMap 
	void Correspondence::printMatchMap(const lpt::ImageFrameGroup& frame_group, const lpt::MatchMap& match_map, string output_file_name) const
	{
		//cout << "printing matchmap..." << endl;
		ofstream fout(output_file_name.c_str());
		int p_id = 0;
		for (int i = 0; i < frame_group.size(); ++i) {
			fout << "------------------------------------------" << endl;
			for (int p = 0; p < frame_group[i].particles.size(); ++p, ++p_id) {
				fout << "Cam " << i << ", " << p << ": " << endl;
				for (int c = 0; c < frame_group.size(); ++c) {
					fout << "\tC" << c << " -- ";
					for (int d = 0; d < NUM_MATCHES; ++d) {
						int index = match_map[p_id][c][d];
						if (index >= 0)
							fout << "[" << index << ", " << p_id << " , " << frame_group[c].particles[index]->id << "]  "; //read below
						//mahdi writing also cumulative particle number, and commenting original below
							//fout << "[" << index << ", " << frame_group[c].particles[index]->id << "]  ";						
						else
							fout << index << "  ";
					}
					fout << endl;
				}
			}
		}
		fout.close();
	}

	void Correspondence::resetMatchMap(lpt::MatchMap& matchmap)
	{
		for (int a = 0; a < matchmap.size(); ++a) {
			for (int b = 0; b < matchmap[b].size(); ++b) {
				for (int c = 0; c < matchmap[a][b].size(); ++c) {
					matchmap[a][b][c] = -1;
				}
			}
		}
	}

	void Correspondence::initializeMatchMap()
	{
		auto& cameras = shared_objects->cameras;
		init_matchmap.resize(cameras.size() * this->initial_max_particles_per_image);
		array<int, NUM_MATCHES> temp;
		for (int z = 0; z < temp.size(); ++z)
			temp[z] = -1;
		for (int a = 0; a < init_matchmap.size(); ++a)
			init_matchmap[a].resize(cameras.size(), temp);
		current_matchmap = init_matchmap;
		lpt::ImageFrameGroup init_framegroup(cameras.size());
		auto pair = std::make_pair(init_framegroup, init_matchmap);
		matchmap_storage.resize(map_storage_size, pair);

		empty_maps_queue.setCapacity(map_storage_size);
		full_maps_queue.setCapacity(map_storage_size);

		for (auto iter = matchmap_storage.begin(); iter != matchmap_storage.end(); ++iter)
			empty_maps_queue.push(iter);
	}
	// similar initializeMatchMap
	void Correspondence::initializeMatchMap(int num_max_particles_per_img, int max_matching)
	{
		cout << " mahdi this is under initializeMatchMap" << endl;
		auto& cameras = shared_objects->cameras;
		init_matchmap.resize(cameras.size() * num_max_particles_per_img);
		array<int, NUM_MATCHES> temp;
		for (int z = 0; z < temp.size(); ++z)
			temp[z] = -1;
		for (int a = 0; a < init_matchmap.size(); ++a)
			init_matchmap[a].resize(cameras.size(), temp);
		current_matchmap = init_matchmap;
		lpt::ImageFrameGroup init_framegroup(cameras.size());
		auto pair = std::make_pair(init_framegroup, init_matchmap);
		matchmap_storage.resize(max_matching, pair);

		empty_maps_queue.setCapacity(max_matching);
		full_maps_queue.setCapacity(max_matching);

		for (auto iter = matchmap_storage.begin(); iter != matchmap_storage.end(); ++iter)
			empty_maps_queue.push(iter);
	}


	void Correspondence::initiate_pairmap() {
		// copy the pairs to unorderd map
		for (int i = 0; i < shared_objects->camera_pairs.size(); i++) {
			string id = to_string(shared_objects->camera_pairs[i].cam_A.id) + to_string(shared_objects->camera_pairs[i].cam_B.id);
			for (int m = 0; m < 3; m++) {
				for (int n = 0; n < 3; n++) {
					pairs_map[id][m][n] = shared_objects->camera_pairs[i].F[m][n];
				}
			}
		}
	}


	void Correspondence::reconstruct_helper(cv::Point3d& p, lpt::Match::Ptr match, vector<lpt::Camera>& cameras) {
		size_t number_of_cams = match->particles.size();

		double** A = new double* [2 * number_of_cams];
		double* B = new double[2 * number_of_cams];
		double* X = new double[3];

		int id = -1;
		for (int i = 0; i < number_of_cams; ++i) {
			int s = i * 2;
			int e = s + 1;
			A[s] = new double[3];
			A[e] = new double[3];
			lpt::ParticleImage::Ptr P = match->particles[i].first;
			size_t camID = match->particles[i].second;
			//cout << " under helper camId = " << camID << endl;//mahdi
			double x = (P->x - cameras[camID].c[0]) / (1. * cameras[camID].f[0]);    // FIXME: May need -1 multiplier for pixel coordinate system (upper left corner origin)
			double y = (P->y - cameras[camID].c[1]) / (1. * cameras[camID].f[1]);    // Convert P.x and P.y to normalized coordinates through intrinsic parameters
			id = P->id;
			A[s][0] = x * cameras[camID].R[2][0] - cameras[camID].R[0][0];
			A[s][1] = x * cameras[camID].R[2][1] - cameras[camID].R[0][1];
			A[s][2] = x * cameras[camID].R[2][2] - cameras[camID].R[0][2];
			A[e][0] = y * cameras[camID].R[2][0] - cameras[camID].R[1][0];
			A[e][1] = y * cameras[camID].R[2][1] - cameras[camID].R[1][1];
			A[e][2] = y * cameras[camID].R[2][2] - cameras[camID].R[1][2];

			B[s] = cameras[camID].T[0] - x * cameras[camID].T[2];
			B[e] = cameras[camID].T[1] - y * cameras[camID].T[2];
		}

		solver.Householder(A, B, number_of_cams * 2, 3);   //Transform A into upper triangular form
		solver.Backsub(A, B, 3, X);
		//array<lpt::Particle3d::float_type, lpt::Particle3d::dim> coords = {{X[0], X[1], X[2]}};

		p.x = X[0];
		p.y = X[1];
		p.z = X[2];

		for (int i = 0; i < number_of_cams * 2; ++i)
			delete[] A[i];
		delete[] A;
		delete[] B;
		delete[] X;
	}

	void Correspondence::output_helper(lpt::Match::Ptr match, ofstream& stream, const lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;
		int height = shared_objects->image_type.rows;
		int width = shared_objects->image_type.cols;
		cout << "mahdi this is under output_helper in correpondence" << endl;

		for (int s = 0; s < match->particles.size() - 1; s++) {
			for (int t = s + 1; t < match->particles.size(); t++) {
				shared_ptr<lpt::ParticleImage> l = match->particles[s].first;
				shared_ptr<lpt::ParticleImage> r = match->particles[t].first;

				int camA_id = match->particles[s].second;
				int camB_id = match->particles[t].second;

				string id = to_string(camA_id) + to_string(camB_id);

				double line[3];

				calculateEpiline(*r, pairs_map[id], line);

				double epipolar_error = calculateEpipolarResidual(line, *l);

				cv::Point3d temp_p;

				lpt::Match::Ptr newmatch = lpt::Match::create();
				newmatch->addParticle(l, (size_t)camA_id);
				newmatch->addParticle(r, (size_t)camB_id);

				reconstruct_helper(temp_p, newmatch, cameras);

				vector<cv::Point3d> temp_object_points(1, temp_p);
				vector<cv::Point2d> temp_image_points(1);

				vector<vector<cv::Point2d>> temp_total_image_points(cameras.size(), temp_image_points);

				for (int i = 0; i < cameras.size(); ++i) {
					size_t camID_temp = (size_t)i;

					cv::Mat R_temp = cv::Mat(3, 3, CV_64F, cameras[camID_temp].R);
					cv::Mat t_vec_temp = cv::Mat(3, 1, CV_64F, cameras[camID_temp].T);
					cv::Mat r_vec_temp = cv::Mat::zeros(3, 1, CV_64F);
					cv::Rodrigues(R_temp, r_vec_temp);

					cv::projectPoints(cv::Mat(temp_object_points), r_vec_temp, t_vec_temp,
						cameras[camID_temp].getCameraMatrix(), cameras[camID_temp].getDistCoeffs(),
						temp_total_image_points[i]);
				}

				// following data can be processed later
				double repo_x_a, repo_y_a, repo_x_b, repo_y_b;

				repo_x_a = temp_total_image_points[camA_id][0].x;
				repo_y_a = temp_total_image_points[camA_id][0].y;

				repo_x_b = temp_total_image_points[camB_id][0].x;
				repo_y_b = temp_total_image_points[camB_id][0].y;

				double x_a_devi, y_a_devi, x_b_devi, y_b_devi;

				x_a_devi = repo_x_a - l->x;
				y_a_devi = repo_y_a - l->y;
				x_b_devi = repo_x_b - r->x;
				y_b_devi = repo_y_b - r->y;

				double repo_a_error, repo_b_error;

				repo_a_error = sqrt(x_a_devi * x_a_devi + y_a_devi * y_a_devi);
				repo_b_error = sqrt(x_b_devi * x_b_devi + y_b_devi * y_b_devi);

				double total_repo_error = repo_a_error + repo_b_error;

				vector<double> temp_intensities(cameras.size(), -1.0);
				vector<double> temp_max_intensities(cameras.size(), -1.0);
				vector<double> temp_mean_intensities(cameras.size(), -1.0);

				int moving_radius = 1;

				for (int i = 0; i < cameras.size(); i++) {
					cv::Point2d temp_repo_p = temp_total_image_points[i][0];

					int temp_repo_x = int(temp_repo_p.x);
					int temp_repo_y = int(temp_repo_p.y);

					double max_intensity = -1.0;
					double mean_intnsity = 0.0;
					int count = 0;
					double temp_pixel_repo_intensity;

					for (int m = temp_repo_x - moving_radius; m <= temp_repo_x + moving_radius; m++) {
						for (int n = temp_repo_y - moving_radius; n <= temp_repo_y + moving_radius; n++) {
							if (m >= 0 && m < width && n >= 0 && n < height) {
								temp_pixel_repo_intensity = (double)frame_group[i].image.at<uchar>(n, m);

								if (m == temp_repo_x && n == temp_repo_y) {
									temp_intensities[i] = temp_pixel_repo_intensity;
								}

								if (temp_pixel_repo_intensity > max_intensity) {
									max_intensity = temp_pixel_repo_intensity;
								}

								mean_intnsity += temp_pixel_repo_intensity;
								count++;

							}
						}
					}

					mean_intnsity = mean_intnsity / (double)count;

					temp_max_intensities[i] = (isnan(max_intensity) ? -1 : max_intensity);
					temp_mean_intensities[i] = (isnan(mean_intnsity) ? -1 : mean_intnsity);
				}

				// output this pair (s, t) to a single line
				// |camA_id|camB_id|
				// |xca|yca|xcb|ycb|
				// |area_a|area_b|
				// |r_a|r_b|
				// |rou_a|rou_b|
				// |peri_a|peri_b|
				// |I_a|I_b|MI_a|MI_b|MEI_a|MEI_b|MI_SHI_xa|MI_SHI_ya|MI_SHI_xb|MI_SHI_yb|
				// |epipolar_error|
				// |X|Y|Z|
				// |xca_repo|yca_repo|xcb_repo|ycb_repo|
				// |repo_error_a|repo_error_b|repo_error_total|repo_SHI_xa|repo_SHI_ya|repo_SHI_xb|repo_SHI_yb|
				// |RI|RMI|RMEI| * Num of cameras

				// |camA_id|camB_id|
				stream << camA_id << " " << camB_id << " ";

				// |xca|yca|xcb|ycb|
				stream << l->x << " " << l->y << " " << r->x << " " << r->y << " ";

				// |area_a|area_b|
				stream << l->area << " " << r->area << " ";

				// |r_a|r_b|
				stream << l->radius << " " << r->radius << " ";

				// |rou_a|rou_b|
				stream << l->roundness << " " << r->roundness << " ";

				// |peri_a|peri_b|
				stream << l->perimeter << " " << r->perimeter << " ";

				// |I_a|I_b|MI_a|MI_b|MEI_a|MEI_b|MI_SHI_xa|MI_SHI_ya|MI_SHI_xb|MI_SHI_yb|
				stream << l->intensity << " " << r->intensity << " ";
				stream << l->max_intensity << " " << r->max_intensity << " ";
				stream << l->mean_intensity << " " << r->mean_intensity << " ";
				stream << l->x_max_shift << " " << l->y_max_shift << " " << r->x_max_shift << " " << r->y_max_shift << " ";

				//////////////////////////////////////////////////////////////////////////////////////

				// Data within this block can be processed later

				// |epipolar error|
				stream << epipolar_error << " ";

				// |X|Y|Z|
				stream << temp_p.x << " " << temp_p.y << " " << temp_p.z << " ";

				// |xca_repo|yca_repo|xcb_repo|ycb_repo|
				stream << repo_x_a << " " << repo_y_a << " " << repo_x_b << " " << repo_y_b << " ";

				// |repo_error_a|repo_error_b|repo_error_total|
				stream << repo_a_error << " " << repo_b_error << " " << total_repo_error << " ";

				// |repo_SHI_xa|repo_SHI_ya|repo_SHI_xb|repo_SHI_yb|
				stream << x_a_devi << " " << y_a_devi << " " << x_b_devi << " " << y_b_devi << " ";

				//////////////////////////////////////////////////////////////////////////////////////////

				// |RI|RMI|RMEI| * Num of cameras
				for (int i = 0; i < cameras.size(); i++) {
					stream << temp_intensities[i] << " ";
					stream << temp_max_intensities[i] << " ";
					stream << temp_mean_intensities[i] << " ";
				}

				stream << "\n";
			}
		}
	}

	// This function is used to to calculate the reprojection error
	bool Correspondence::reprojection_error(lpt::Match::Ptr match, double threshold, const lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;
		auto& pairs = shared_objects->camera_pairs;

		auto& S = this->shared_objects->S;
		auto& P = this->shared_objects->P;
		double temp_x, temp_y, temp_z;

		vector<cv::Point3d> object_points(1);
		vector<cv::Point2d> image_points(1);

		//cout << " mahdi under reprrj erro we have match->particles.size() " << match->particles.size() << endl;
		vector<vector<cv::Point2d>> total_image_points(match->particles.size(), image_points);

		int height = shared_objects->image_type.rows;
		int width = shared_objects->image_type.cols;

		size_t number_of_cams = match->particles.size();
		if (number_of_cams >= 2) {
			double empty[3] = { 0 };
			double empty1 = 0;

			cv::Point3d p;

			reconstruct_helper(p, match, cameras);

			object_points[0] = p;
			//cout << fixed << setprecision(2);
			//cout << fixed << setprecision(4) << " mahdi point p after reconstruct helper p= " << p.x << " , " << p.y << " , " << p.z << endl;
			for (int i = 0; i < match->particles.size(); ++i) {
				//cout << " mahdi under reprrj erro we have int i = 0; i < match->particles.size(); ++i " << i << endl;
				size_t camID = match->particles[i].second;

				cv::Mat R = cv::Mat(3, 3, CV_64F, cameras[camID].R);
				cv::Mat t_vec = cv::Mat(3, 1, CV_64F, cameras[camID].T);
				cv::Mat r_vec = cv::Mat::zeros(3, 1, CV_64F);
				cv::Rodrigues(R, r_vec);
				// mahdi priting the contents of R, r_vec
				/*cout << "-------------------------start---------------------------" << endl;
				cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
				fmt->set64fPrecision(4);
				fmt->set32fPrecision(4);
				cout << "R:\n" << fmt->format(R) << endl;
				cout << "r_vec:\n" << fmt->format(r_vec) << endl;
				cout << "-----------------------------end-----------------------" << endl;
				*/// mahdi priting the contents of R, r_vec
				cv::Mat zero_dist_coeff = cv::Mat::zeros(1, 4, CV_32F); // mahdi passing zero instead of distort coeff
				//cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[camID].getCameraMatrix(), cameras[camID].getDistCoeffs(), total_image_points[i]);
				cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[camID].getCameraMatrix(), zero_dist_coeff, total_image_points[i]);
			}
			//// mahdi priting the contents of image points
			//cout << "-------------------------total_image_points start---------------------------" << endl;
			//for (int i = 0; i < total_image_points.size(); i++)
			//{
			//	cout << "Particle " << i + 1 << ":" << endl;
			//	for (int j = 0; j < total_image_points[i].size(); j++)
			//	{
			//		std::cout << "Point " << j + 1 << ": (" << total_image_points[i][j].x << ", " << total_image_points[i][j].y << ")" << std::endl;
			//	}
			//}
			//cout << "----------------------------- total_image_points end-----------------------" << endl;
			//// mahdi priting the contents of R, r_vec
			// calculate the reprojection error
			double error = 0.0;
			double temp_error = 0.0;
			double match_x, match_y, repo_x, repo_y;
			//cout << "-------------------------diff between coordinates we have---------------------------" << endl;

			for (int i = 0; i < match->particles.size(); i++) {
				match_x = match->particles[i].first->x;
				match_y = match->particles[i].first->y;
				size_t camID = match->particles[i].second; // mahdi

				repo_x = total_image_points[i][0].x;
				repo_y = total_image_points[i][0].y;

				//cout << setprecision(4) << " p & camID = " << i << camID << endl;
				//cout << setprecision(4) << " error x= " << match_x - repo_x << " match x = " << match_x << " and repoj x = " << repo_x << endl;
				//cout << setprecision(4) << " error y= " << match_y - repo_y << " match y = " << match_y << " and repoj y = " << repo_y << endl;


				temp_error = sqrt((match_x - repo_x) * (match_x - repo_x) + (match_y - repo_y) * (match_y - repo_y));
				error += temp_error;
			}
			//cout << "-------------------------end diff between coordinates we have---------------------------" << endl;
			error = error / (double)match->particles.size(); // mahdi: divides error by 4, 3 or 2
			//cout << fixed << setprecision(4) << " error is = " << error << endl;
			//return true;
			if (error < threshold) {
				if (COLLECT_DATA)
					output_helper(match, positiveData, frame_group);

				return true;
			}
			else {
				return false;
			}

		}
	}

	//changes by Mahdi
	vector <MatchAndRpjErrs> Correspondence::findMinErrMatch(const vector < MatchAndRpjErrs>& matchAndErr) {
		vector < MatchAndRpjErrs> minErrMatches;
		if (matchAndErr.empty()) {
			//throw invalid_argument("matchAndErr is empty");
			//cerr << "matchAndErr is empty\n";
			return minErrMatches;
		}
		//definitions
		map<int, vector < MatchAndRpjErrs>> matchGroup;

		for (auto it = matchGroup.begin(); it != matchGroup.end(); ++it) {
			vector<MatchAndRpjErrs>& vec = it->second;
			// Now you can access elements in the vector
			MatchAndRpjErrs& firstElement = vec[0];
			cout << fixed << setprecision(4) << " MatchAndRpjErrs& firstElement = vec[0]; " << firstElement.p3drec << endl;
		}

		int groupId = 0;
		double thresholdDistance = 5.0;
		//cout << "before Loop A";

		/*for (const auto& match : matchAndErr) {
			cout << "Loop A" << endl;
			bool added = false;
			for (auto& group : matchGroup) {
				cout << " Loop B and inside min finder and outside if " << endl;
				cout << "  group.second[0].p3drec= " << group.second[0].p3drec << " match.p3drec= " << match.p3drec << " norm= " << cv::norm(group.second[0].p3drec - match.p3drec) << endl;

				if (cv::norm(group.second[0].p3drec - match.p3drec) < thresholdDistance) {
					cout << " cluster found, group.second[0].p3drec= " << group.second[0].p3drec << " match.p3drec= " << match.p3drec << " norm= " << cv::norm(group.second[0].p3drec - match.p3drec) << endl;
					group.second.push_back(match);
					added = true;
					break;
				}
			}
			if (!added) {
				cout << " new match gourp created " << endl;
				matchGroup[groupId++] = { match };
			}
		}*/


		for (const auto& match : matchAndErr) {
			bool added = false;
			for (auto& group : matchGroup) {
				//cout << "////////// inside min finder MatchReturn/////// cv norm= " << cv::norm(group.second[0].p3drec - match.p3drec) << endl;

				if (cv::norm(group.second[0].p3drec - match.p3drec) < thresholdDistance) {
					group.second.push_back(match);
					added = true;
					break;
				}
			}
			if (!added) {
				matchGroup[groupId++] = { match };
			}
		}

		for (const auto& group : matchGroup) {
			//cout << "Loop C";

			auto minErrIt = min_element(group.second.begin(), group.second.end(),
				[](const MatchAndRpjErrs& a, const MatchAndRpjErrs& b) {
					return a.reprojectionErr < b.reprojectionErr;
				}
			);
			minErrMatches.push_back(*minErrIt);
		}
		return minErrMatches;
	}

	vector <lpt::Match::Ptr> Correspondence::findMinErrMatchReturn(const vector < MatchAndRpjErrs>& matchAndErr, double thresholdDistance) {
		//mahdi passing matcher because of accessing matcher->params.threshold_distance only
		vector < lpt::Match::Ptr> minErrMatches;
		if (matchAndErr.empty()) {
			//throw invalid_argument("matchAndErr is empty");
			//cout << "matchAndErr is empty" << endl;
			return minErrMatches;
		}



		//definitions
		map<int, vector < MatchAndRpjErrs>> matchGroup;
		int groupId = 0;
		//double thresholdDistance = 10;		
		//for (const auto& match : matchAndErr) {
		//	bool added = false;
		//	for (auto& group : matchGroup) {
		//		//cout << "////////// inside min finder MatchReturn/////// cv norm= " << cv::norm(group.second[0].p3drec - match.p3drec) << endl;

		//		if (cv::norm(group.second[0].p3drec - match.p3drec) < thresholdDistance) {
		//			group.second.push_back(match);
		//			added = true;
		//			break;
		//		}
		//	}
		//	if (!added) {
		//		matchGroup[groupId++] = { match };
		//	}
		//}

		for (const auto& match : matchAndErr) {
			//if (match.p3drec.z > 0)				
			//	return minErrMatches;
			//cout << " under new correspond we have match.p3drec.z=" << match.p3drec.z << endl;
			//cout << "Loop A" << endl;
			bool added = false;
			for (auto& group : matchGroup) {
				//cout << " Loop B and inside min finder and outside if " << endl;
				//cout << "  group.second[0].p3drec= " << group.second[0].p3drec << " match.p3drec= " << match.p3drec << " norm= " << cv::norm(group.second[0].p3drec - match.p3drec) << endl;

				if (cv::norm(group.second[0].p3drec - match.p3drec) < thresholdDistance) {
					//cout << " cluster found, group.second[0].p3drec= " << group.second[0].p3drec << " match.p3drec= " << match.p3drec << " norm= " << cv::norm(group.second[0].p3drec - match.p3drec) << endl;
					group.second.push_back(match);
					added = true;
					break;
				}
			}
			if (!added) {
				//cout << " new match gourp created " << endl;
				matchGroup[groupId++] = { match };
			}
		}

		for (const auto& group : matchGroup) {
			auto minErrIt = min_element(group.second.begin(), group.second.end(),
				[](const MatchAndRpjErrs& a, const MatchAndRpjErrs& b) {
					return a.reprojectionErr < b.reprojectionErr;
				}
			);
			minErrMatches.push_back(minErrIt->match);
			//cout << fixed << setprecision(4) << "\n minErrMatch rpj erro= " << minErrIt->reprojectionErr << " & P3d= " << minErrIt->p3drec << " & cams= ";
			//for (const auto& particlepair : minErrIt->match->particles)
				//cout << particlepair.second << " ";
		}
		return minErrMatches;
	}




	//function below by mahdi
	void Correspondence::reprojection_error_calc(lpt::Match::Ptr match, cv::Point3d& p3d, double& rpj_err) {
		auto& cameras = shared_objects->cameras;
		auto& pairs = shared_objects->camera_pairs;

		auto& S = this->shared_objects->S;
		auto& P = this->shared_objects->P;
		double temp_x, temp_y, temp_z;

		vector<cv::Point3d> object_points(1);
		vector<cv::Point2d> image_points(1);

		//cout << " mahdi under reprrj erro we have match->particles.size() " << match->particles.size() << endl;
		vector<vector<cv::Point2d>> total_image_points(match->particles.size(), image_points);

		int height = shared_objects->image_type.rows;
		int width = shared_objects->image_type.cols;

		size_t number_of_cams = match->particles.size();
		if (number_of_cams >= 2) {
			double empty[3] = { 0 };
			double empty1 = 0;

			cv::Point3d p;

			reconstruct_helper(p, match, cameras);

			object_points[0] = p;
			p3d = p;
			//cout << fixed << setprecision(2);			
			for (int i = 0; i < match->particles.size(); ++i) {
				//cout << " mahdi under reprrj erro we have int i = 0; i < match->particles.size(); ++i " << i << endl;
				size_t camID = match->particles[i].second;

				cv::Mat R = cv::Mat(3, 3, CV_64F, cameras[camID].R);
				cv::Mat t_vec = cv::Mat(3, 1, CV_64F, cameras[camID].T);
				cv::Mat r_vec = cv::Mat::zeros(3, 1, CV_64F);
				cv::Rodrigues(R, r_vec);
				// mahdi priting the contents of R, r_vec
				/*cout << "-------------------------start---------------------------" << endl;
				cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
				fmt->set64fPrecision(4);
				fmt->set32fPrecision(4);
				cout << "R:\n" << fmt->format(R) << endl;
				cout << "r_vec:\n" << fmt->format(r_vec) << endl;
				cout << "-----------------------------end-----------------------" << endl;
				*/// mahdi priting the contents of R, r_vec
				cv::Mat zero_dist_coeff = cv::Mat::zeros(1, 4, CV_32F); // mahdi passing zero instead of distort coeff
				//cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[camID].getCameraMatrix(), cameras[camID].getDistCoeffs(), total_image_points[i]);
				cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[camID].getCameraMatrix(), zero_dist_coeff, total_image_points[i]);
			}
			//// mahdi priting the contents of image points
			//cout << "-------------------------total_image_points start---------------------------" << endl;
			//for (int i = 0; i < total_image_points.size(); i++)
			//{
			//	cout << "Particle " << i + 1 << ":" << endl;
			//	for (int j = 0; j < total_image_points[i].size(); j++)
			//	{
			//		std::cout << "Point " << j + 1 << ": (" << total_image_points[i][j].x << ", " << total_image_points[i][j].y << ")" << std::endl;
			//	}
			//}
			//cout << "----------------------------- total_image_points end-----------------------" << endl;
			//// mahdi priting the contents of R, r_vec
			// calculate the reprojection error
			double error = 0.0;
			double temp_error = 0.0;
			double match_x, match_y, repo_x, repo_y;
			//cout << "-------------------------inside reprojection check---------------------------" << endl;
			//cout << " mahdi point p after reconstruct helper p= " << p.x << " , " << p.y << " , " << p.z << endl;
			for (int i = 0; i < match->particles.size(); i++) {
				match_x = match->particles[i].first->x;
				match_y = match->particles[i].first->y;
				size_t camID = match->particles[i].second; // mahdi

				repo_x = total_image_points[i][0].x;
				repo_y = total_image_points[i][0].y;

				/*cout << " p & camID = " << i << camID << endl;
				cout << " error x= " << match_x - repo_x << " match x = " << match_x << " and repoj x = " << repo_x << endl;
				cout << " error y= " << match_y - repo_y << " match y = " << match_y << " and repoj y = " << repo_y << endl;*/


				temp_error = sqrt((match_x - repo_x) * (match_x - repo_x) + (match_y - repo_y) * (match_y - repo_y));
				error += temp_error;
			}
			//cout << "-------------------------end inside reprojection check---------------------------" << endl;
			error = error / (double)match->particles.size(); // mahdi: divides error by 4, 3 or 2
			//cout << " error is = " << error << endl;
			rpj_err = error;
		}
	}
	//end changes by Mahdi
	void Correspondence::construct_pair_feature(vector<lpt::Match::Ptr>& match_vec, const lpt::ImageFrameGroup& frame_group, vector<vector<float>>& temp_space) {
		int match_size = match_vec.size();
		//cout << "mahdi this is under construct_pair_feature under correpo" << endl;
		if (match_size == 0)
			return;

		int num_features = 0;//pca_rows;

		vector<float> temp_vec(num_features, 0.0f);

		auto& cameras = shared_objects->cameras;
		int height = shared_objects->image_type.rows;
		int width = shared_objects->image_type.cols;

		int camA_id, camB_id;
		shared_ptr<lpt::ParticleImage> l, r;
		string id;
		double line[3];
		float epipolar_error;

		cv::Point3d temp_p;

		vector<cv::Point3d> temp_object_points(1, temp_p);
		vector<cv::Point2d> temp_image_points(1);
		vector<vector<cv::Point2d>> temp_total_image_points(num_grayscale_cameras, temp_image_points);

		cv::Mat R_temp;
		cv::Mat t_vec_temp;
		cv::Mat r_vec_temp;

		float repo_x_a, repo_y_a, repo_x_b, repo_y_b;
		float x_a_devi, y_a_devi, x_b_devi, y_b_devi;
		float repo_a_error, repo_b_error, total_repo_error;

		int moving_radius = 1;

		vector<float> temp_intensities(num_grayscale_cameras, -1.0);
		vector<float> temp_max_intensities(num_grayscale_cameras, -1.0);
		vector<float> temp_mean_intensities(num_grayscale_cameras, -1.0);

		for (int mi = 0; mi < match_size; mi++) {
			l = match_vec[mi]->particles[0].first;
			r = match_vec[mi]->particles[1].first;

			camA_id = match_vec[mi]->particles[0].second;
			camB_id = match_vec[mi]->particles[1].second;

			id = to_string(camA_id) + to_string(camB_id);
			calculateEpiline(*r, pairs_map[id], line);
			epipolar_error = calculateEpipolarResidual(line, *l);

			reconstruct_helper(temp_object_points[0], match_vec[mi], cameras);

			// reproject to the original cameras
			for (int i = 0; i < match_vec[mi]->particles.size(); ++i) {
				//size_t camID_temp = (size_t)grayscale_camera_ids[i];
				size_t camID_temp = (size_t)match_vec[mi]->particles[i].second;

				R_temp = cv::Mat(3, 3, CV_64F, cameras[camID_temp].R);
				t_vec_temp = cv::Mat(3, 1, CV_64F, cameras[camID_temp].T);
				r_vec_temp = cv::Mat::zeros(3, 1, CV_64F);
				cv::Rodrigues(R_temp, r_vec_temp);

				cv::projectPoints(cv::Mat(temp_object_points), r_vec_temp, t_vec_temp,
					cameras[camID_temp].getCameraMatrix(), cameras[camID_temp].getDistCoeffs(),
					temp_total_image_points[i]);
			}

			repo_x_a = temp_total_image_points[0][0].x;
			repo_y_a = temp_total_image_points[0][0].y;

			repo_x_b = temp_total_image_points[1][0].x;
			repo_y_b = temp_total_image_points[1][0].y;

			x_a_devi = repo_x_a - l->x;
			y_a_devi = repo_y_a - l->y;
			x_b_devi = repo_x_b - r->x;
			y_b_devi = repo_y_b - r->y;

			repo_a_error = sqrt(x_a_devi * x_a_devi + y_a_devi * y_a_devi);
			repo_b_error = sqrt(x_b_devi * x_b_devi + y_b_devi * y_b_devi);
			total_repo_error = repo_a_error + repo_b_error;

			// Here need to project to the grayscale cameras
			for (int i = 0; i < num_grayscale_cameras; i++) {
				size_t camID_temp = (size_t)grayscale_camera_ids[i];
				R_temp = cv::Mat(3, 3, CV_64F, cameras[camID_temp].R);
				t_vec_temp = cv::Mat(3, 1, CV_64F, cameras[camID_temp].T);
				r_vec_temp = cv::Mat::zeros(3, 1, CV_64F);
				cv::Rodrigues(R_temp, r_vec_temp);

				cv::projectPoints(cv::Mat(temp_object_points), r_vec_temp, t_vec_temp,
					cameras[camID_temp].getCameraMatrix(), cameras[camID_temp].getDistCoeffs(),
					temp_total_image_points[i]);

				cv::Point2d temp_repo_p = temp_total_image_points[i][0];

				int temp_repo_x = int(temp_repo_p.x);
				int temp_repo_y = int(temp_repo_p.y);

				float max_intensity = -1.0;
				float mean_intnsity = 0.0;
				int count = 0;
				float temp_pixel_repo_intensity;

				for (int m = temp_repo_x - moving_radius; m <= temp_repo_x + moving_radius; m++) {
					for (int n = temp_repo_y - moving_radius; n <= temp_repo_y + moving_radius; n++) {
						if (m >= 0 && m < width && n >= 0 && n < height) {
							uchar pixel_intensity_uchar = frame_group[grayscale_camera_ids[i]].image.at<uchar>(n, m);

							temp_pixel_repo_intensity = (float)pixel_intensity_uchar;

							if (m == temp_repo_x && n == temp_repo_y) {
								temp_intensities[i] = temp_pixel_repo_intensity;

								//cv::circle(frame_group[grayscale_camera_ids[i]].image, temp_repo_p, 5, 255, 1);
							}

							if (temp_pixel_repo_intensity > max_intensity) {
								max_intensity = temp_pixel_repo_intensity;
							}

							mean_intnsity += temp_pixel_repo_intensity;
							count++;

						}
					}
				}

				//cv::imwrite("C:/LPT/LPT-HighDensity-nn/VPTV-high_density/data/output/temppppppp.jpg", frame_group[grayscale_camera_ids[i]].image);

				mean_intnsity = mean_intnsity / (float)count;

				temp_max_intensities[i] = (isnan(max_intensity) ? -1 : max_intensity);
				temp_mean_intensities[i] = (isnan(mean_intnsity) ? -1 : mean_intnsity);
			}


			int count = 0;

			// |camA_id|camB_id|
			temp_vec[count++] = (float)camA_id;
			temp_vec[count++] = (float)camB_id;

			// |xca|yca|xcb|ycb|
			temp_vec[count++] = l->x;
			temp_vec[count++] = l->y;
			temp_vec[count++] = r->x;
			temp_vec[count++] = r->y;

			// |area_a|area_b|
			temp_vec[count++] = l->area;
			temp_vec[count++] = r->area;

			// |r_a|r_b|
			temp_vec[count++] = l->radius;
			temp_vec[count++] = r->radius;

			// |rou_a|rou_b|
			temp_vec[count++] = l->roundness;
			temp_vec[count++] = r->roundness;

			// |epipolar error|
			temp_vec[count++] = epipolar_error;

			//// |X|Y|Z|
			//temp_vec[13] = temp_p.x;
			//temp_vec[14] = temp_p.y;
			//temp_vec[15] = temp_p.z;

			// |xca_repo|yca_repo|xcb_repo|ycb_repo|
			temp_vec[count++] = repo_x_a;
			temp_vec[count++] = repo_y_a;
			temp_vec[count++] = repo_x_b;
			temp_vec[count++] = repo_y_b;

			// |repo_error_a|repo_error_b|repo_error_total|
			temp_vec[count++] = repo_a_error;
			temp_vec[count++] = repo_b_error;
			temp_vec[count++] = total_repo_error;

			// |repo_SHI_xa|repo_SHI_ya|repo_SHI_xb|repo_SHI_yb|
			temp_vec[count++] = x_a_devi;
			temp_vec[count++] = y_a_devi;
			temp_vec[count++] = x_b_devi;
			temp_vec[count++] = y_b_devi;

			// |RI|RMI|RMEI| * Num of grayscale cameras
			for (int i = 0; i < num_grayscale_cameras; i++) {
				temp_vec[count++] = temp_intensities[i];
				temp_vec[count++] = temp_max_intensities[i];
				temp_vec[count++] = temp_mean_intensities[i];
			}

			temp_space.push_back(temp_vec);
		}
	}

	void Correspondence::construct_test_image_data(vector<lpt::Match::Ptr>& matches, ofstream& stream, const lpt::ImageFrameGroup& frame_group) {
		auto& cameras = shared_objects->cameras;
		int num_cameras = cameras.size();
		int match_size = matches.size();

		int max_images = 20 > match_size ? match_size : 20;

		string folder_name = "C:/LPT/LPT-HighDensity-nn/VPTV-high_density/data/output/particle_images/";

		if (match_size == 0)
			return;

		vector<cv::Point3d> temp_object_points(match_size, cv::Point3d());
		vector<cv::Point2d> temp_image_points(match_size, cv::Point2d());
		vector<vector<cv::Point2d>> temp_total_image_points(num_cameras, temp_image_points);

		cv::Mat R_temp;
		cv::Mat t_vec_temp;
		cv::Mat r_vec_temp;

		lpt::Match::Ptr temp_match;

		int height = shared_objects->image_type.rows;
		int width = shared_objects->image_type.cols;

		int moving_radius = 5;

		for (int mi = 0; mi < matches.size(); mi++) {
			temp_match = matches[mi];
			reconstruct_helper(temp_object_points[mi], temp_match, cameras);
		}

		for (int i = 0; i < num_cameras; ++i) {
			size_t camID_temp = i;

			R_temp = cv::Mat(3, 3, CV_64F, cameras[camID_temp].R);
			t_vec_temp = cv::Mat(3, 1, CV_64F, cameras[camID_temp].T);
			r_vec_temp = cv::Mat::zeros(3, 1, CV_64F);
			cv::Rodrigues(R_temp, r_vec_temp);

			cv::projectPoints(cv::Mat(temp_object_points), r_vec_temp, t_vec_temp,
				cameras[camID_temp].getCameraMatrix(), cameras[camID_temp].getDistCoeffs(),
				temp_total_image_points[i]);
		}

		for (int mi = 0; mi < matches.size(); mi++) {
			if (mi == max_images)
				break;

			temp_match = matches[mi];

			cv::Point3d temp_3d_point = temp_object_points[mi];

			//stream << temp_3d_point.x << " ";
			//stream << temp_3d_point.y << " ";
			//stream << temp_3d_point.z << " ";

			for (int i = 0; i < num_cameras; ++i) {
				size_t camID_temp = i;
				cv::Point2d temp_repo_p = temp_total_image_points[i][mi];

				int temp_repo_x = int(temp_repo_p.x);
				int temp_repo_y = int(temp_repo_p.y);

				if (temp_repo_x - moving_radius >= 0 && temp_repo_x + moving_radius < width && temp_repo_y - moving_radius >= 0 && temp_repo_y + moving_radius < height) {
					cv::Rect rect(temp_repo_x - moving_radius, temp_repo_y - moving_radius, moving_radius * 2, moving_radius * 2);
					cv::Mat roi = frame_group[i].image(rect);

					string img_name = folder_name + to_string(temp_3d_point.x) + " " + to_string(temp_3d_point.y) + " " + to_string(temp_3d_point.z) + " " + to_string(temp_repo_x) + " " + to_string(temp_repo_y) + " " + to_string(i) + ".jpg";

					cv::imwrite(img_name, roi);
				}

				//for (int m = temp_repo_x - moving_radius; m <= temp_repo_x + moving_radius; m++) {
				//	for (int n = temp_repo_y - moving_radius; n <= temp_repo_y + moving_radius; n++) {
				//		if (m >= 0 && m < width && n >= 0 && n < height) {
				//			int pixel_intensity = (int)frame_group[i].image.at<uchar>(n, m);

				//			stream << i << " ";
				//			stream << m << " ";
				//			stream << n << " ";
				//			stream << pixel_intensity << " ";
				//		}
				//	}
				//}

			}

			//stream << "\n";

		}
	}


	PointMatcher::PointMatcher()
	{
		cout << "Epipolar Point matcher constructed" << endl;
		//positiveData.open("../../../data/output/positive_data.txt");
		//negativeData.open("../../../data/output/negative_data.txt");

	}

	PointMatcher::~PointMatcher() {
		positiveData.close();
		negativeData.close();
	}

	void PointMatcher::initialize()
	{
		this->initializeMatchMap();
	}

	void PointMatcher::initialize(int num_max_particles_per_img, int max_matching)
	{
		this->initializeMatchMap(num_max_particles_per_img, max_matching);
	}

	void PointMatcher::addControls()
	{
		void* matcher_void_ptr = static_cast<void*> (this);
		cv::createTrackbar("Match Threshold correspond", string(), &params.match_thresh_level, 100, callbackMatchThresh, matcher_void_ptr);
	}


	void PointMatcher::findEpipolarMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap)
	{
		cout << " mahdi this is under findfpipolarmatches under correspond in cpp" << endl;
		auto& camera_pairs = shared_objects->camera_pairs;
		size_t match_overload = 0;
		vector<size_t> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();

		//cout << " mahdi length of num_particles= " << frame_group.size() << " num_particles[0]= " << num_particles[0] << endl;

		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		//for (int i = 1; i < frame_group.size(); ++i) //mahdi
			//cout << " mahdi vector num_particles element i = " << i << " value = " << num_particles[i] << " frame_group[i].particles.size()= " << frame_group[i].particles.size() << endl;

		for (vector<lpt::CameraPair>::iterator pair = camera_pairs.begin(); pair != camera_pairs.end(); ++pair) {
			//cout << " mahdi match_overload= " << match_overload << endl;


			int cam_a = pair->cam_A.id;
			int cam_b = pair->cam_B.id;
			//cout << " mahdi for the current pair cam_a = " << cam_a << " and cam_b = " << cam_b << endl;
			const vector<ParticleImage::Ptr>& cam_A_particles = frame_group[cam_a].particles;
			const vector<ParticleImage::Ptr>& cam_B_particles = frame_group[cam_b].particles;


			size_t b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
			size_t b_end = num_particles[cam_b];

			size_t a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			size_t a_end = num_particles[cam_a];

			//cout << " mahdi b_start= " << b_start << " b_end= " << b_end << " a_start= " << a_start << " a_end= " << a_end << endl;

			for (size_t b_id = b_start; b_id < b_end; ++b_id) {
				double lineA[3];
				calculateEpiline(*cam_B_particles[b_id - b_start], pair->F, lineA);
				//cout << "mahdi inner loop in cam B current b_id = " << b_id << " now printing A" << endl;
				//for (const auto& e : lineA) { // uncomment for delay in processing count down
				//	cout << e << " " << endl;
				//}
				for (size_t a_id = a_start; a_id < a_end; ++a_id) {
					double residual = calculateEpipolarResidual(lineA, *cam_A_particles[a_id - a_start]);

					//cout << "mahdi match thresh level real = " << params.match_threshold << endl; // mahdi
					if (residual <= params.match_threshold) {
						//cout << "mahdi residual is maller than threshold = " << residual << endl;
						auto itb = std::find(matchmap[b_id][cam_a].begin(), matchmap[b_id][cam_a].end(), -1);
						auto ita = std::find(matchmap[a_id][cam_b].begin(), matchmap[a_id][cam_b].end(), -1);

						//cout << " matchmap[a_id][cam_b].begin() = " << matchmap[a_id][cam_b][0] << endl;
						if (itb != matchmap[b_id][cam_a].end() && ita != matchmap[a_id][cam_b].end()) {
							*itb = static_cast<int>(a_id - a_start);
							*ita = static_cast<int>(b_id - b_start);
							//loop 1 to 10 over matchmap[b_id][cam_a]
						}
						else
							match_overload++;
					}
				}
			}
		}
		//cout << match_overload << endl;
	}

	void PointMatcher::findUniqueMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches)
	{
		cout << " mahdi this is under finduniquematches in cpp " << endl; // this is never called
		vector<size_t> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		size_t num_cameras = frame_group.size();
		matches.clear();

		for (size_t cam_a = 0; cam_a < num_cameras - 3; ++cam_a)
		{
			size_t a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a)
			{
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched)
				{
					cout << " mahdi under is4wayMatched under findunique" << endl; // this is never called
					for (size_t cam_b = cam_a + 1; cam_b < num_cameras - 2; ++cam_b)
					{
						size_t b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
						for (size_t match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
							int b = matchmap[a + a_start][cam_b][match_ab];
							if (b < 0)
								break;
							lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];

							if (!Pb->is_4way_matched)
							{
								for (size_t cam_c = cam_b + 1; cam_c < num_cameras - 1; ++cam_c)
								{
									size_t c_start = (cam_c != 0 ? num_particles[cam_c - 1] : 0);
									for (int match_bc = 0; match_bc < NUM_MATCHES; ++match_bc)
									{
										int c = matchmap[b + b_start][cam_c][match_bc];
										if (c < 0)
											break;

										lpt::ParticleImage::Ptr Pc = frame_group[cam_c].particles[c];

										if (!Pc->is_4way_matched && std::count(matchmap[a + a_start][cam_c].begin(), matchmap[a + a_start][cam_c].end(), c))
										{
											for (size_t cam_d = cam_c + 1; cam_d < num_cameras; ++cam_d)
											{
												size_t d_start = (cam_d != 0 ? num_particles[cam_d - 1] : 0);
												for (int match_cd = 0; match_cd < NUM_MATCHES; ++match_cd)
												{
													int d = matchmap[c + c_start][cam_d][match_cd];
													if (d < 0)
														break;
													lpt::ParticleImage::Ptr Pd = frame_group[cam_d].particles[d];
													if (!Pd->is_4way_matched && std::count(matchmap[a + a_start][cam_d].begin(), matchmap[a + a_start][cam_d].end(), d) && std::count(matchmap[b + b_start][cam_d].begin(), matchmap[b + b_start][cam_d].end(), d))
													{
														lpt::Match::Ptr newmatch = lpt::Match::create();
														newmatch->addParticle(Pa, cam_a);
														newmatch->addParticle(Pb, cam_b);
														newmatch->addParticle(Pc, cam_c);
														newmatch->addParticle(Pd, cam_d);

														if (!reprojection_error(newmatch, REPROJECTION_ERROR, frame_group))
															continue;

														// output to the negative data, there are supposed to be 4 * 3 / 2 = 6 pairs
														 // first find negative data between cam_a and cam_b

														if (COLLECT_DATA) {
															// particle a and particle b
															// find the negative particle in cam_b
															for (size_t match_ab_temp = 0; match_ab_temp < NUM_MATCHES; ++match_ab_temp) {
																int b_temp = matchmap[a + a_start][cam_b][match_ab_temp];

																if (b_temp < 0)
																	break;

																if (b_temp != b) {
																	lpt::ParticleImage::Ptr Pb_temp = frame_group[cam_b].particles[b_temp];

																	lpt::Match::Ptr newmatch_ab_temp = lpt::Match::create();

																	newmatch_ab_temp->addParticle(Pa, cam_a);
																	newmatch_ab_temp->addParticle(Pb_temp, cam_b);

																	output_helper(newmatch_ab_temp, negativeData, frame_group);

																}
															}
															// second cam_a and cam_c
															for (size_t match_ac_temp = 0; match_ac_temp < NUM_MATCHES; ++match_ac_temp) {
																int c_temp = matchmap[a + a_start][cam_c][match_ac_temp];

																if (c_temp < 0)
																	break;

																if (c_temp != c) {
																	lpt::ParticleImage::Ptr Pc_temp = frame_group[cam_c].particles[c_temp];

																	lpt::Match::Ptr newmatch_ac_temp = lpt::Match::create();

																	newmatch_ac_temp->addParticle(Pa, cam_a);
																	newmatch_ac_temp->addParticle(Pc_temp, cam_c);

																	output_helper(newmatch_ac_temp, negativeData, frame_group);

																}
															}
															// third cam_a and cam_d
															for (size_t match_ad_temp = 0; match_ad_temp < NUM_MATCHES; ++match_ad_temp) {
																int d_temp = matchmap[a + a_start][cam_d][match_ad_temp];

																if (d_temp < 0)
																	break;

																if (d_temp != d) {
																	lpt::ParticleImage::Ptr Pd_temp = frame_group[cam_d].particles[d_temp];

																	lpt::Match::Ptr newmatch_ad_temp = lpt::Match::create();

																	newmatch_ad_temp->addParticle(Pa, cam_a);
																	newmatch_ad_temp->addParticle(Pd_temp, cam_d);

																	output_helper(newmatch_ad_temp, negativeData, frame_group);

																}
															}
															// cam_b and cam_c
															for (size_t match_bc_temp = 0; match_bc_temp < NUM_MATCHES; ++match_bc_temp) {
																int c_temp_b = matchmap[b + b_start][cam_c][match_bc_temp];

																if (c_temp_b < 0)
																	break;

																if (c_temp_b != c) {
																	lpt::ParticleImage::Ptr Pc_temp_b = frame_group[cam_c].particles[c_temp_b];

																	lpt::Match::Ptr newmatch_bc_temp = lpt::Match::create();

																	newmatch_bc_temp->addParticle(Pb, cam_b);
																	newmatch_bc_temp->addParticle(Pc_temp_b, cam_c);

																	output_helper(newmatch_bc_temp, negativeData, frame_group);

																}
															}
															// cam_b and cam_d
															for (size_t match_bd_temp = 0; match_bd_temp < NUM_MATCHES; ++match_bd_temp) {
																int d_temp_b = matchmap[b + b_start][cam_d][match_bd_temp];

																if (d_temp_b < 0)
																	break;

																if (d_temp_b != d) {
																	lpt::ParticleImage::Ptr Pd_temp_b = frame_group[cam_d].particles[d_temp_b];

																	lpt::Match::Ptr newmatch_bd_temp = lpt::Match::create();

																	newmatch_bd_temp->addParticle(Pb, cam_b);
																	newmatch_bd_temp->addParticle(Pd_temp_b, cam_d);

																	output_helper(newmatch_bd_temp, negativeData, frame_group);

																}
															}
															// cam_c and cam_d
															for (size_t match_cd_temp = 0; match_cd_temp < NUM_MATCHES; ++match_cd_temp) {
																int d_temp_c = matchmap[c + c_start][cam_d][match_cd_temp];

																if (d_temp_c < 0)
																	break;

																if (d_temp_c != d) {
																	lpt::ParticleImage::Ptr Pd_temp_c = frame_group[cam_d].particles[d_temp_c];

																	lpt::Match::Ptr newmatch_cd_temp = lpt::Match::create();

																	newmatch_cd_temp->addParticle(Pc, cam_c);
																	newmatch_cd_temp->addParticle(Pd_temp_c, cam_d);

																	output_helper(newmatch_cd_temp, negativeData, frame_group);

																}
															}
														}

														matches.push_back(std::move(newmatch));
														Pa->is_4way_matched = true;
														Pb->is_4way_matched = true;
														Pc->is_4way_matched = true;
														Pd->is_4way_matched = true;
														match_ab = NUM_MATCHES;
														match_bc = NUM_MATCHES;
														match_cd = NUM_MATCHES;
														cam_b = num_cameras;
														cam_c = num_cameras;
														cam_d = num_cameras;
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
			}
		}
	}

	void PointMatcher::find3WayMatches_mahdi(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
		// mahdi: this is by Yu, and I only modified the one using CUDA
		cout << " under PointMatcher find3way matches_wo_clear" << endl;
		vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		//cout << " mahdi under find3way matches_wo_clear" << endl;
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		int num_cameras = frame_group.size();
		//cout << " mahdi under find3way matches_wo_clear, num cameras = " << num_cameras << endl;
		for (int cam_a = 0; cam_a < frame_group.size() - 2; ++cam_a) {
			//cout << " mahdi under for loop in wo clear, cam_a= " << cam_a << endl;
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) { //mahdi loop over partciles in cam_a
				lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
				if (!Pa->is_4way_matched) {
					//cout << " mahdi under woclear called is 4way matched" << endl;
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

											if (!reprojection_error(newmatch, REPROJECTION_ERROR, frame_group))
												continue;

											if (COLLECT_DATA) {
												// particle a and particle b
												// find the negative particle in cam_b
												for (size_t match_ab_temp = 0; match_ab_temp < NUM_MATCHES; ++match_ab_temp) {
													int b_temp = matchmap[a + a_start][cam_b][match_ab_temp];

													if (b_temp < 0)
														break;

													if (b_temp != b) {
														lpt::ParticleImage::Ptr Pb_temp = frame_group[cam_b].particles[b_temp];

														lpt::Match::Ptr newmatch_ab_temp = lpt::Match::create();

														newmatch_ab_temp->addParticle(Pa, cam_a);
														newmatch_ab_temp->addParticle(Pb_temp, cam_b);

														output_helper(newmatch_ab_temp, negativeData, frame_group);

													}
												}
												// second cam_a and cam_c
												for (size_t match_ac_temp = 0; match_ac_temp < NUM_MATCHES; ++match_ac_temp) {
													int c_temp = matchmap[a + a_start][cam_c][match_ac_temp];

													if (c_temp < 0)
														break;

													if (c_temp != c) {
														lpt::ParticleImage::Ptr Pc_temp = frame_group[cam_c].particles[c_temp];

														lpt::Match::Ptr newmatch_ac_temp = lpt::Match::create();

														newmatch_ac_temp->addParticle(Pa, cam_a);
														newmatch_ac_temp->addParticle(Pc_temp, cam_c);

														output_helper(newmatch_ac_temp, negativeData, frame_group);

													}
												}

												// cam_b and cam_c
												for (size_t match_bc_temp = 0; match_bc_temp < NUM_MATCHES; ++match_bc_temp) {
													int c_temp_b = matchmap[b + b_start][cam_c][match_bc_temp];

													if (c_temp_b < 0)
														break;

													if (c_temp_b != c) {
														lpt::ParticleImage::Ptr Pc_temp_b = frame_group[cam_c].particles[c_temp_b];

														lpt::Match::Ptr newmatch_bc_temp = lpt::Match::create();

														newmatch_bc_temp->addParticle(Pb, cam_b);
														newmatch_bc_temp->addParticle(Pc_temp_b, cam_c);

														output_helper(newmatch_bc_temp, negativeData, frame_group);

													}
												}
											}

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

		if (COLLECT_IMG_DATA)
			construct_test_image_data(matches, imageData, frame_group);
	}

	//void Correspondence::find2WayMathces(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
	//	vector<int> num_particles(frame_group.size());
	//	num_particles[0] = frame_group[0].particles.size();
	//	for (int i = 1; i < frame_group.size(); ++i)
	//		num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];
	//
	//	int num_cameras = frame_group.size();
	//
	//	for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
	//		int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
	//		for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
	//			lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
	//			if (!Pa->is_4way_matched) {
	//				for (int cam_b = cam_a + 1; cam_b < frame_group.size(); ++cam_b) {
	//					int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
	//					for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
	//						int b = matchmap[a + a_start][cam_b][match_ab];
	//						if (b < 0)
	//							break;
	//						lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];
	//
	//						lpt::Match::Ptr newmatch = lpt::Match::create();
	//
	//						vector<vector<float>> temp_data;
	//						vector<lpt::Match::Ptr> temp_pairs_vec;
	//						vector<float> temp_transformed_data(10, 0);
	//
	//						newmatch->addParticle(Pa, cam_a);
	//						newmatch->addParticle(Pb, cam_b);
	//
	//						temp_pairs_vec.push_back(newmatch);
	//						construct_pair_feature(temp_pairs_vec, frame_group, temp_data);
	//
	//						for (int i = 0; i < 30; i++) {
	//							temp_data[0][i] = (temp_data[0][i] - x_train_mean[i]) / x_train_std[i];
	//						}
	//
	//						// temp_data * pca_paras
	//						for (int i = 0; i < 10; i++) {
	//							for (int j = 0; j < 30; j++) {
	//								temp_transformed_data[i] += temp_data[0][j] * pca_parameters[j][i];
	//							}
	//						}
	//
	//						float prob = 0.0f; float temp = 0.0f;
	//						// perform LR prob
	//						for (int i = 0; i < 10; i++) {
	//							temp += temp_transformed_data[i] * lr_data[i];
	//						}
	//
	//						temp += lr_data[10];
	//
	//						prob = 1 / (1 + exp(-temp));
	//
	//						if (prob > 0.8) {
	//							matches.push_back(std::move(newmatch));
	//
	//							Pa->is_4way_matched = true;
	//							Pb->is_4way_matched = true;
	//
	//							match_ab = NUM_MATCHES;
	//							cam_b = num_cameras;
	//						}
	//
	//
	//					}
	//				}
	//			}
	//		}
	//	}
	//
	//}
	//
	//void PointMatcher::find2WayMathces_wo_clear(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
	//	vector<int> num_particles(frame_group.size());
	//	num_particles[0] = frame_group[0].particles.size();
	//	for (int i = 1; i < frame_group.size(); ++i)
	//		num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];
	//
	//	int num_cameras = frame_group.size();
	//
	//	vector<vector<float>> temp_data;
	//	vector<torch::jit::IValue> inputs;
	//	vector<lpt::Match::Ptr> temp_pairs_vec;
	//
	//	for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
	//		int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
	//		for (int a = 0; a < frame_group[cam_a].particles.size(); ++a) {
	//			lpt::ParticleImage::Ptr Pa = frame_group[cam_a].particles[a];
	//			if (!Pa->is_4way_matched) {
	//				for (int cam_b = cam_a + 1; cam_b < frame_group.size(); ++cam_b) {
	//					int b_start = (cam_b != 0 ? num_particles[cam_b - 1] : 0);
	//					for (int match_ab = 0; match_ab < NUM_MATCHES; ++match_ab) { //loop through all A,B matches
	//						int b = matchmap[a + a_start][cam_b][match_ab];
	//						if (b < 0)
	//							break;
	//						lpt::ParticleImage::Ptr Pb = frame_group[cam_b].particles[b];
	//
	//						lpt::Match::Ptr newmatch = lpt::Match::create();
	//						newmatch->addParticle(Pa, cam_a);
	//						newmatch->addParticle(Pb, cam_b);
	//
	//						temp_pairs_vec.push_back(newmatch);
	//
	//					}
	//				}
	//			}
	//		}
	//	}
	//
	//	construct_pair_feature(temp_pairs_vec, frame_group, temp_data);
	//
	//	int rows = temp_data.size();
	//
	//	if (rows != 0) {
	//		int cols = temp_data[0].size();
	//		auto options = torch::TensorOptions().dtype(at::kFloat);
	//
	//		if (cols) {
	//			at::Tensor temp_data_tensor = torch::ones({ rows, cols }, options);
	//
	//			for (int i = 0; i < rows; i++)
	//				temp_data_tensor.slice(0, i, i + 1) = torch::from_blob(temp_data[i].data(), { cols }, options);
	//
	//			temp_data_tensor.to(at::kCUDA);
	//			temp_data_tensor = (temp_data_tensor - mean_tensor) / std_tensor;
	//			temp_data_tensor = at::matmul(temp_data_tensor, pca_tensor).to(at::kCUDA);
	//
	//			//cout << temp_data_tensor << endl;
	//
	//			vector<torch::jit::IValue> inputs;
	//			inputs.push_back(temp_data_tensor);
	//
	//			torch_module.eval();
	//			torch::NoGradGuard no_grad;
	//
	//			at::Tensor output = torch_module.forward(inputs).toTensor();
	//			output = torch::sigmoid(output);
	//			output = torch::round(output);
	//
	//			output = output.to(at::kCPU);
	//
	//			auto output_a = output.accessor<float, 2>();
	//
	//			int temp_count = 0;
	//			for (int i = 0; i < output_a.size(0); i++) {
	//				if ((int)output_a[i][0] == 1) {
	//
	//					if (!temp_pairs_vec[i]->particles[0].first->is_4way_matched && !temp_pairs_vec[i]->particles[1].first->is_4way_matched) {
	//						matches.push_back(temp_pairs_vec[i]);
	//						temp_pairs_vec[i]->particles[0].first->is_4way_matched = true;
	//						temp_pairs_vec[i]->particles[1].first->is_4way_matched = true;
	//					}
	//
	//					temp_count++;
	//				}
	//			}
	//		}
	//	}
	//}

	void PointMatcher::find2WayMathces_wo_clear(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
		vector<int> num_particles(frame_group.size());
		num_particles[0] = frame_group[0].particles.size();
		cout << " mahdi under find2way matches_wo_clear in cpp" << endl;
		for (int i = 1; i < frame_group.size(); ++i)
			num_particles[i] = frame_group[i].particles.size() + num_particles[i - 1];

		int num_cameras = frame_group.size();
		cout << " mahdi under find2way matches_wo_clear, num cameras = " << num_cameras << endl;

		for (int cam_a = 0; cam_a < frame_group.size() - 1; ++cam_a) {
			int a_start = (cam_a != 0 ? num_particles[cam_a - 1] : 0);
			cout << " mahdi under for loop in 2way wo clear, cam_a= " << cam_a << endl;
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
							matches.push_back(std::move(newmatch));
						}
					}
				}
			}
		}
	}

	void PointMatcher::find3WayMatches(const lpt::ImageFrameGroup& frame_group, lpt::MatchMap& matchmap, vector<lpt::Match::Ptr>& matches) {
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
	}

	void PointMatcher::removeNonUniqueMatches(vector<Match::Ptr>& matches)
	{
		for (int i = 0; i < matches.size(); ++i) {
			if (matches[i]->isUnique() == false) {
				matches[i].reset();
			}
		}
	}

	//void PointMatcher::find3wayMatches(tuple<int,int,int>& triple, lpt::ImageFrameGroup& cameragroup, vector<lpt::Match::Ptr>& matches) {
	//	int camA = std::get<0>(triple);
	//	int camB = std::get<1>(triple);
	//	int camC = std::get<2>(triple);
	//	vector<lpt::Match::Ptr> matches3way;
	//	for (int i = 0; i < cameragroup[ camA ].particles.size(); ++i) {
	//		lpt::ParticleImage* Pa = cameragroup[camA].particles[i].get();
	//		set<lpt::ParticleImage*>& matchesAB = Pa->matches2way[camB];
	//		set<lpt::ParticleImage*>::iterator matchAB;
	//		if( ! Pa->is_3way_matched ) 
	//		for (matchAB = matchesAB.begin(); matchAB != matchesAB.end(); ++matchAB){
	//			ParticleImage* Pb = (*matchAB);
	//			set<ParticleImage*>& matchesBC = Pb->matches2way[camC];
	//			set<ParticleImage*>& matchesAC = Pa->matches2way[camC];
	//			set<ParticleImage*>::iterator matchAC;
	//			if( ! Pb->is_3way_matched ) 
	//			for (matchAC = matchesAC.begin(); matchAC != matchesAC.end(); ++matchAC) {
	//				ParticleImage* Pc = (*matchAC);
	//				if( ! Pc->is_3way_matched ) 
	//				if( matchesBC.count(Pc) ) { 
	//					lpt::Match::Ptr newmatch = lpt::Match::create();
	//					newmatch->addParticle(Pa,camA);
	//					newmatch->addParticle(Pb,camB);
	//					newmatch->addParticle(Pc,camC);
	//					matches3way.push_back(newmatch);
	//					Pa->is_3way_matched = true;
	//					Pb->is_3way_matched = true;
	//					Pc->is_3way_matched = true;
	//					matchAB = matchesAB.end();
	//					matchAC = matchesAC.end();
	//				}
	//			}
	//		}
	//	}
	//	removeNonUniqueMatches( matches3way );
	//	std::move(matches3way.begin(), matches3way.end(), std::back_inserter(matches) );
	//}

	//void PointMatcher::find4WayMatches(tuple<int,int,int,int>& quad, lpt::ImageFrameGroup& cameragroup, vector<lpt::Match::Ptr>& matches) {
	//	int camA = std::get<0>(quad);
	//	int camB = std::get<1>(quad);
	//	int camC = std::get<2>(quad);
	//	int camD = std::get<3>(quad);
	//	vector<lpt::Match::Ptr> matches4way;
	//	for (int i = 0; i < cameragroup[ camA ].particles.size(); ++i) {
	//		lpt::ParticleImage* Pa = cameragroup[camA].particles[i].get();
	//		set<lpt::ParticleImage*>& matchesAB = Pa->matches2way[camB];
	//		set<lpt::ParticleImage*>::iterator matchAB;
	//		if( ! Pa->is_4way_matched ) 
	//		for(matchAB = matchesAB.begin(); matchAB != matchesAB.end(); ++matchAB) { //loop through all A,B matches
	//			lpt::ParticleImage* Pb = (*matchAB);
	//			set<lpt::ParticleImage*>& matchesBC = Pb->matches2way[camC];
	//			set<lpt::ParticleImage*>::iterator matchBC;
	//			if( ! Pb->is_4way_matched ) 
	//			for (matchBC = matchesBC.begin(); matchBC != matchesBC.end(); ++matchBC) {
	//				lpt::ParticleImage* Pc = (*matchBC);
	//				set<lpt::ParticleImage*>& matchesCD = Pc->matches2way[camD];
	//				set<lpt::ParticleImage*>::iterator matchCD;
	//				if( ! Pc->is_4way_matched ) 
	//				for (matchCD = matchesCD.begin(); matchCD != matchesCD.end(); ++matchCD) {
	//					lpt::ParticleImage* Pd = (*matchCD);
	//					if( ! Pd->is_4way_matched ) 
	//					if(Pd->matches2way[camA].count(Pa) && Pb->matches2way[camD].count(Pd) && Pa->matches2way[camC].count(Pc)) {
	//						if(! Pa->is_4way_matched && ! Pb->is_4way_matched && ! Pc->is_4way_matched && ! Pd->is_4way_matched) { 
	//							lpt::Match::Ptr newmatch = lpt::Match::create();
	//							newmatch->addParticle(Pa,camA);
	//							newmatch->addParticle(Pb,camB);
	//							newmatch->addParticle(Pc,camC);
	//							newmatch->addParticle(Pd,camD);
	//							matches4way.push_back(std::move(newmatch));
	//							Pa->is_4way_matched = true;
	//							Pb->is_4way_matched = true;
	//							Pc->is_4way_matched = true;
	//							Pd->is_4way_matched = true;
	//							matchAB = matchesAB.end();
	//							matchBC = matchesBC.end();
	//							matchCD = matchesCD.end();
	//						}
	//					} 
	//				}
	//			}
	//		}
	//	}
	//	removeNonUniqueMatches( matches4way );
	//	std::move(matches4way.begin(), matches4way.end(), std::back_inserter(matches) );
	//}

	Reconstruct3D::Reconstruct3D() {
		//reprojectionFile.open("../../../data/output/reprojection.txt");
		//reprojection_intensity_File.open("../../../data/output/reprojection_intensity.txt");
	}

	Reconstruct3D::~Reconstruct3D()
	{
		/*
		for (size_t i = 0; i < positions.size(); i++) {
			double x = positions[i][0] / frame_count;
			double y = positions[i][1] / frame_count;
			double z = positions[i][2] / frame_count;
			axis << i << "\t" << x << "\t" << y << "\t" << z << endl;
		}
		reprojectionFile.close();
		reprojection_intensity_File.close();
		cout << "***************************succeed to write reprojection.txt*****************************" << endl;

		cout << "Reconstruct3D destructed" << endl;
		axis.close();*/
	}

	struct myclass {
		bool operator() (lpt::Particle3d_Ptr p1, lpt::Particle3d_Ptr p2)
		{
			return (p1->X[1] < p2->X[1]);
		}
	} mycomp;

	//void Reconstruct3D::calculateReprojectionError(vector<lpt::Match::Ptr>& matches, const lpt::Frame3d& frame) {
	//	auto& cameras = shared_objects->cameras;
	//	vector<cv::Point3d> object_points(frame.objects.size());
	//	vector<cv::Point2d> image_points(frame.objects.size());
	//	vector<double> intensities(frame.objects.size(), 0);
	//	vector<double> radii(frame.objects.size(), 0);

	//	vector<vector<cv::Point2d>> total_image_points(cameras.size(), image_points);
	//	vector<vector<double>> total_intensities(cameras.size(), intensities);
	//	vector<vector<double>> total_radii(cameras.size(), radii);

	//	// The cameras to omit
	//	vector<int> camerasID_omit;

	//	int img_height = this->shared_objects->image_type.rows;
	//	int img_width = this->shared_objects->image_type.cols;

	//	auto particles = frame.objects;
	//	auto& S = this->shared_objects->S;
	//	auto& P = this->shared_objects->P;
	//	double temp_x, temp_y, temp_z;

	//	for (int p = 0; p < frame.objects.size(); ++p) {
	//		temp_x = frame.objects[p]->X[0];
	//		temp_y = frame.objects[p]->X[1];
	//		temp_z = frame.objects[p]->X[2];

	//		object_points[p].x = S[0][0] * (temp_x + P[0]) + S[1][0] * (temp_y + P[1]) + S[2][0] * (temp_z + P[2]);
	//		object_points[p].y = S[0][1] * (temp_x + P[0]) + S[1][1] * (temp_y + P[1]) + S[2][1] * (temp_z + P[2]);
	//		object_points[p].z = S[0][2] * (temp_x + P[0]) + S[1][2] * (temp_y + P[1]) + S[2][2] * (temp_z + P[2]);
	//	}		
	//	for (int i = 0; i < frame.camera_frames.size(); ++i) {
	//		cv::Mat R = cv::Mat(3, 3, CV_64F, cameras[i].R);
	//		cv::Mat t_vec = cv::Mat(3, 1, CV_64F, cameras[i].T);
	//		cv::Mat r_vec = cv::Mat::zeros(3, 1, CV_64F);
	//		cv::Rodrigues(R, r_vec);

	//		cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[i].getCameraMatrix(), cameras[i].getDistCoeffs(), total_image_points[i]);

	//		int temp_xc, temp_yc;
	//		int radius = 1;

	//		//// find the max intensity of surrounding particle image
	//		//for (int j = 0; j < total_image_points[i].size(); j++) {
	//		//	temp_xc = (int)total_image_points[i][j].x;
	//		//	temp_yc = (int)total_image_points[i][j].y;

	//		//	int max = -1; // numeric_limits<int>::min();
	//		//	int temp_intensity;

	//		//	for (int m = -radius; m <= radius; m++) {
	//		//		for (int n = -radius; n <= radius; n++) {
	//		//			if (temp_xc + n >= 0 && temp_xc + n < img_width && temp_yc + m >= 0 && temp_yc + m < img_height) {
	//		//				temp_intensity = (int)frame.camera_frames[i].image.at<uchar>(temp_yc + m, temp_xc + n);

	//		//				if (temp_intensity > max)
	//		//					max = temp_intensity;
	//		//			}
	//		//		}
	//		//	}

	//		//	total_intensities[i][j] = max;
	//		//}

	//		// find the avg intensity of surrounding particle image
	//		for (int j = 0; j < total_image_points[i].size(); j++) {
	//			temp_xc = (int)total_image_points[i][j].x;
	//			temp_yc = (int)total_image_points[i][j].y;

	//			int count = 0; // numeric_limits<int>::min();
	//			int temp_intensity = 0;

	//			for (int m = -radius; m <= radius; m++) {
	//				for (int n = -radius; n <= radius; n++) {
	//					if (temp_xc + n >= 0 && temp_xc + n < img_width && temp_yc + m >= 0 && temp_yc + m < img_height) {
	//						temp_intensity += (double)frame.camera_frames[i].image.at<uchar>(temp_yc + m, temp_xc + n);
	//						count++;
	//					}
	//				}
	//			}

	//			total_intensities[i][j] = (count != 0 ? (double)temp_intensity / (double)count : -1.0);
	//		}
	//	}

	//	for (int p = 0; p < object_points.size(); p++) {
	//		// [X, Y, Z] [0, x0, y0, xp0, yp0] ...... [i, xi, yi, xpi, ypi]
	//		reprojectionFile << to_string(object_points[p].x) << " " << to_string(object_points[p].y) << " " << to_string(object_points[p].z) << " ";
	//		reprojection_intensity_File << to_string(object_points[p].x) << " " << to_string(object_points[p].y) << " " << to_string(object_points[p].z) << " ";

	//		for (int i = 0; i < matches[p]->particles.size(); i++) {
	//			int camId = matches[p]->particles[i].second;

	//			// output the camera ID
	//			reprojectionFile << to_string(camId) << " ";
	//			reprojection_intensity_File << to_string(camId) << " ";

	//			// output the image coordinates (matches and reprojection) to the reprojectionFile
	//			reprojectionFile << to_string(matches[p]->particles[i].first->x) << " " << to_string(matches[p]->particles[i].first->y) << " ";
	//			reprojectionFile << to_string(total_image_points[camId][p].x) << " " << to_string(total_image_points[camId][p].y) << " ";

	//			reprojection_intensity_File << to_string(total_intensities[camId][p]) << " " << matches[p]->particles[i].first->radius << " ";
	//		}

	//		reprojectionFile << "\n";
	//		reprojection_intensity_File << "\n";
	//	}

	//	//for (int p = 0; p < image_points.size(); ++p) {
	//	//	//cout << "Projected x: " << image_points[p].x << "; y: " << image_points[p].y << endl;
	//	//	reprojectionFile << to_string(object_points[p].x) << " " << to_string(object_points[p].y) << " " << to_string(object_points[p].z) << " ";

	//	//	reprojectionFile << to_string(i) << " ";

	//	//	//cout << "Camera : " << i << ". #p: " << p << "/" << object_points.size() << ". matches size: " << matches.size() << endl;
	//	//	//cout << "Match particles size: " << matches[p]->particles.size() << endl;

	//	//	//cout << endl;

	//	//	//cout << "Matches   x: " << matches[p]->particles[i].first->x << "; y: " << matches[p]->particles[i].first->y << endl;

	//	//	//string matchX = to_string(matches[p]->particles[i].first->x);
	//	//	//string matchY = to_string(matches[p]->particles[i].first->y);

	//	//	//string matchX = to_string(matches[p]->particles[i].first->x);
	//	//	//string matchY = to_string(matches[p]->particles[i].first->y);

	//	//	//reprojectionFile << matchX << " " << matchY << " ";

	//	//	//reprojectionFile << to_string(image_points[p].x) << " " << to_string(image_points[p].y) << "\n";
	//	//}



	//}

	void Reconstruct3D::reconstruct3DFrame(vector<lpt::Match::Ptr>& matches, lpt::Frame3d& frame)
	{
		//std::cout << "mahdi under reconstruct 3D frame without SVD" << endl;
		//system("pause");
		auto& cameras = shared_objects->cameras;
		int ii = 0; //mahdi
		//cout << "mahdi before loop we have matches.size()= " << matches.size() << endl;
		for (int m = 0; m < matches.size(); ++m) {
			ii++;
			if (matches[m]) {
				size_t number_of_cams = matches[m]->particles.size();
				//if (ii != 0) { std::cout << "mahdi matches.size ()=" << matches.size() << " loop counter=" << ii << " number_of_cams = " << number_of_cams << endl; } //mahdi			
				if (number_of_cams >= 2) {
					double empty[3] = { 0 };
					double empty1 = 0;
					double** A = new double* [2 * number_of_cams];
					double* B = new double[2 * number_of_cams];
					double* X = new double[3];

					int id = -1;
					for (int i = 0; i < number_of_cams; ++i) {
						//cout << "mahdi under inner loop, printing i = " << i << endl; //mahdi
						int s = i * 2;
						int e = s + 1;
						A[s] = new double[3];
						A[e] = new double[3];
						lpt::ParticleImage::Ptr P = matches[m]->particles[i].first;
						size_t camID = matches[m]->particles[i].second;
						//cout << "mahdi cam Id is = " << camID << endl;
						//cout << "camera params for the bove cam id are cx = " << cameras[camID].c[0] << "and cy = " << cameras[camID].c[1] << endl;
						//cout << "camera params for the bove cam id are fx = " << cameras[camID].f[0] << "and fy = " << cameras[camID].f[1] << endl;
						double x = (P->x - cameras[camID].c[0]) / (1. * cameras[camID].f[0]);    // FIXME: May need -1 multiplier for pixel coordinate system (upper left corner origin)
						double y = (P->y - cameras[camID].c[1]) / (1. * cameras[camID].f[1]);    // Convert P.x and P.y to normalized coordinates through intrinsic parameters
						//cout << "x and y values under loop tpe double = " << x << " & " << y << endl;
						//cout << "px and py values under loop tpe double = " << P->x << " & " << P->y << endl;
						//system("pause");
						id = P->id;
						//cout << "mahdi p id is = " << id << endl;
						A[s][0] = x * cameras[camID].R[2][0] - cameras[camID].R[0][0];
						A[s][1] = x * cameras[camID].R[2][1] - cameras[camID].R[0][1];
						A[s][2] = x * cameras[camID].R[2][2] - cameras[camID].R[0][2];
						A[e][0] = y * cameras[camID].R[2][0] - cameras[camID].R[1][0];
						A[e][1] = y * cameras[camID].R[2][1] - cameras[camID].R[1][1];
						A[e][2] = y * cameras[camID].R[2][2] - cameras[camID].R[1][2];

						B[s] = cameras[camID].T[0] - x * cameras[camID].T[2];
						B[e] = cameras[camID].T[1] - y * cameras[camID].T[2];
					}

					////mahdi printing A and B
					//std::cout << std::fixed;
					//std::cout << std::setprecision(6);
					//cout << "printing A" << endl;
					//for (int i = 0; i < 2 * number_of_cams; i++) {
					//	for (int j = 0; j < 3; j++) {
					//		//cout << " i= " << i << ", j= " << j <<  " A[i][j] = " << A[i][j] << " ";
					//		cout << A[i][j] << " ";
					//	}
					//	cout << endl;
					//}
					//cout << "printing B" << endl;
					////mahdi printing A and B
					//for (int i = 0; i < 2 * number_of_cams; i++) {
					//	cout << B[i] << " ";
					//	cout << endl;
					//}
					////

					solver.Householder(A, B, number_of_cams * 2, 3);   //Transform A into upper triangular form

					//cout << "printing A Householder" << endl;
					//for (int i = 0; i < 2 * number_of_cams; i++) {
					//	for (int j = 0; j < 3; j++) {
					//		//cout << " i= " << i << ", j= " << j <<  " A[i][j] = " << A[i][j] << " ";
					//		cout << A[i][j] << " ";
					//	}
					//	cout << endl;
					//}

					solver.Backsub(A, B, 3, X);
					//cout << "printing X" << endl;
					////mahdi printing A and B
					//for (int i = 0; i < 3; i++) {
					//	cout << X[i] << " ";
					//	cout << endl;
					//}

					//array<lpt::Particle3d::float_type, lpt::Particle3d::dim> coords = {{X[0], X[1], X[2]}};
					lpt::Particle3d_Ptr newparticle = lpt::Particle3d::create();
					newparticle->id = id;
					if (this->shared_objects->isRotation_Correction) {
						auto& S = this->shared_objects->S;
						auto& P = this->shared_objects->P;
						/*
						newparticle->X[0] -= P[0];
						newparticle->X[1] -= P[1];
						newparticle->X[2] -= P[2];

						newparticle->X[0] = S[0][0] * X[0] + S[0][1] * X[1] + S[0][2] * X[2];
						newparticle->X[1] = S[1][0] * X[0] + S[1][1] * X[1] + S[1][2] * X[2];
						newparticle->X[2] = S[2][0] * X[0] + S[2][1] * X[1] + S[2][2] * X[2];
						*/
						newparticle->X[0] = S[0][0] * X[0] + S[0][1] * X[1] + S[0][2] * X[2] - P[0];
						newparticle->X[1] = S[1][0] * X[0] + S[1][1] * X[1] + S[1][2] * X[2] - P[1];
						newparticle->X[2] = S[2][0] * X[0] + S[2][1] * X[1] + S[2][2] * X[2] - P[2];
					}
					else {
						newparticle->X[0] = X[0];
						newparticle->X[1] = X[1];
						newparticle->X[2] = X[2];
					}
					newparticle->frame_index = frame.frame_index;

					frame.objects.push_back(std::move(newparticle));
					//draw(frame);//mahdi
					for (int i = 0; i < number_of_cams * 2; ++i)
						delete[] A[i];
					delete[] A;
					delete[] B;
					delete[] X;
				}
			}
		}
	}

	void Reconstruct3D::draw(lpt::Frame3d& frame)
	{
		auto& cameras = shared_objects->cameras;
		vector<cv::Point3d> object_points(frame.objects.size());
		vector<cv::Point2d> image_points(frame.objects.size());
		auto particles = frame.objects;

		//if (positions.size() != frame.objects.size())
		//	positions.resize(frame.objects.size());

		std::sort(particles.begin(), particles.end(), mycomp);

		auto& S = this->shared_objects->S;
		auto& P = this->shared_objects->P;
		double temp_x, temp_y, temp_z;
		for (int p = 0; p < frame.objects.size(); ++p) {
			//object_points[p].x = frame.objects[p]->X[0] + this->shared_objects->P[0];
			//object_points[p].y = frame.objects[p]->X[1] + this->shared_objects->P[1];
			//object_points[p].z = frame.objects[p]->X[2] + this->shared_objects->P[2];
			temp_x = frame.objects[p]->X[0];
			temp_y = frame.objects[p]->X[1];
			temp_z = frame.objects[p]->X[2];

			object_points[p].x = S[0][0] * (temp_x + P[0]) + S[1][0] * (temp_y + P[1]) + S[2][0] * (temp_z + P[2]);
			object_points[p].y = S[0][1] * (temp_x + P[0]) + S[1][1] * (temp_y + P[1]) + S[2][1] * (temp_z + P[2]);
			object_points[p].z = S[0][2] * (temp_x + P[0]) + S[1][2] * (temp_y + P[1]) + S[2][2] * (temp_z + P[2]);
			//cout << " mahd frame.objects has x = " << temp_x << " , y= " << temp_y << " , z= " << temp_z << endl;
			//cout << fixed;
			cout << setprecision(6);
			cout << " mahdi object point vis3d has x = " << object_points[p].x << " , y= " << object_points[p].y << " , z= " << object_points[p].z << endl;

			//for (int i = 0; i < frame.objects[p]->X.size(); i++) {
			//	positions[p][i] += particles[p]->X[i];
			//}
		}

		//frame_count++;

		for (int i = 0; i < frame.camera_frames.size(); ++i) {
			//if (frame.camera_frames[i].image.channels() == 1)
			//	cv::cvtColor(frame.camera_frames[i].image, frame.camera_frames[i].image, CV_GRAY2BGR);
			cv::Mat R = cv::Mat(3, 3, CV_64F, cameras[i].R);
			cv::Mat t_vec = cv::Mat(3, 1, CV_64F, cameras[i].T);
			cv::Mat r_vec = cv::Mat::zeros(3, 1, CV_64F);
			cv::Rodrigues(R, r_vec);

			cv::projectPoints(cv::Mat(object_points), r_vec, t_vec, cameras[i].getCameraMatrix(), cameras[i].getDistCoeffs(), image_points);

			for (int p = 0; p < image_points.size(); ++p) {
				cv::circle(frame.camera_frames[i].image, image_points[p], 5, 255, 1);
				cv::circle(frame.camera_frames[i].image, image_points[p], 2, 0, -1);
				cout << " projected image point cam#" << i << " has values = " << image_points[p] << endl;
			}

			//cv::namedWindow("temp", cv::WINDOW_AUTOSIZE);
			//cv::imshow("temp", frame.camera_frames[i].image);
			//cv::waitKey(0);
		}

		//cv::destroyAllWindows();
	}

	Reconstruct3DwithSVD::Reconstruct3DwithSVD() : Reconstruct3D() {}

	Reconstruct3DwithSVD::~Reconstruct3DwithSVD() {}

	void Reconstruct3DwithSVD::reconstruct3DFrame(vector<lpt::Match::Ptr>& matches, lpt::Frame3d& frame)
	{
		cout << "Mahdi Under 3D reconstruct with SVD" << endl; // this is never called
		auto& cameras = shared_objects->cameras;
		for (int m = 0; m < matches.size(); ++m) {
			if (matches[m]) {
				size_t number_of_cams = matches[m]->particles.size();
				if (number_of_cams >= 2) {
					cv::Mat A = cv::Mat::zeros(2 * static_cast<int>(cameras.size()), 3, CV_64F);
					cv::Mat B = cv::Mat::zeros(2 * static_cast<int>(cameras.size()), 1, CV_64F);
					cv::Mat X = cv::Mat::zeros(3, 1, CV_64F);
					lpt::Particle3d_Ptr newparticle = lpt::Particle3d::create();

					int id = -1;
					for (int i = 0; i < number_of_cams; ++i) {
						int s = i * 2;
						int e = s + 1;

						lpt::ParticleImage::Ptr P = matches[m]->particles[i].first;
						size_t cam_id = matches[m]->particles[i].second;
						newparticle->camera_ids[i] = cam_id;								//WARNING this will cause a segfault if number_of_cameras is greater than four;
						id = P->id;

						double x = (P->x - cameras[cam_id].c[0]) / cameras[cam_id].f[0];
						double y = (P->y - cameras[cam_id].c[1]) / cameras[cam_id].f[1];

						A.at<double>(s, 0) = x * cameras[cam_id].R[2][0] - cameras[cam_id].R[0][0];
						A.at<double>(s, 1) = x * cameras[cam_id].R[2][1] - cameras[cam_id].R[0][1];
						A.at<double>(s, 2) = x * cameras[cam_id].R[2][2] - cameras[cam_id].R[0][2];

						A.at<double>(e, 0) = y * cameras[cam_id].R[2][0] - cameras[cam_id].R[1][0];
						A.at<double>(e, 1) = y * cameras[cam_id].R[2][1] - cameras[cam_id].R[1][1];
						A.at<double>(e, 2) = y * cameras[cam_id].R[2][2] - cameras[cam_id].R[1][2];

						B.at<double>(s) = cameras[cam_id].T[0] - x * cameras[cam_id].T[2];
						B.at<double>(e) = cameras[cam_id].T[1] - y * cameras[cam_id].T[2];
					}

					svd(A);
					svd.backSubst(B, X);

					newparticle->id = id;
					newparticle->X[0] = X.at<double>(0);
					newparticle->X[1] = X.at<double>(1);
					newparticle->X[2] = X.at<double>(2);
					newparticle->frame_index = frame.frame_index;

					double max_svalue = 0, min_svalue = 0;
					cv::minMaxIdx(svd.w, &min_svalue, &max_svalue);
					double cond_A = max_svalue / min_svalue;
					double normL2_A = max_svalue;
					double normL2_B = cv::norm(B);
					double normL2_X = cv::norm(X);

					cv::Mat AX = A * X;
					cv::Mat R = B - AX;
					double normL2_R = cv::norm(R);
					double theta = atan(normL2_R / cv::norm(AX));


					//for(int i = 0; i < cameras.size()+10000; ++i) {
					//	
					//	/*double x_u2 = 0; 
					//	double y_u2 = 0;

					//	lpt::ParticleImage* P = matches[m]->particles[i].first;
					//	int cam_id = matches[m]->particles[i].second;
					//	
					//	lpt::Camera& cam = cameras[cam_id];
					//	cv::Mat camera_matrix = cam.getCameraMatrix();
					//	cv::Mat rotation_matrix(3,3, CV_64F, cam.R); 		
					//	cv::Mat translation_vec(3,1, CV_64F, cam.T);
					//	cv::Mat dist_coeffs = cam.getDistCoeffs();

					//	cv::Mat rotation_vec;
					//	cv::Rodrigues(rotation_matrix, rotation_vec);

					//	cv::Point3d point;
					//	point.x = newparticle->X[0];
					//	point.x = newparticle->X[1];
					//	point.x = newparticle->X[2];

					//	vector<cv::Point3d> point3D; 
					//	point3D.push_back(point);
					//	vector<cv::Point2d> image_point_distorted;
					//	vector<cv::Point2d> image_point;*/
					//	/*cv::projectPoints( cv::Mat(point3D), rotation_vec, translation_vec, camera_matrix, dist_coeffs, image_point_distorted );
					//	cv::undistortPoints( image_point_distorted, image_point, camera_matrix, dist_coeffs, cv::Mat(), camera_matrix);

					//	for(int dist_id = 0; dist_id < 4; ++dist_id) {

					//		cv::Mat dist_coeffs_perturbed = dist_coeffs.clone();
					//		dist_coeffs_perturbed.at<double>(dist_id) += cam.dist_coeffs_u[dist_id];

					//		vector<cv::Point2d> image_point_perturbed;

					//		cv::undistortPoints( image_point_distorted, image_point_perturbed, camera_matrix, dist_coeffs_perturbed, cv::Mat(), camera_matrix);

					//		double x_dxprime = 0;
					//		double y_dyprime = 0;

					//		if ( cam.dist_coeffs_u[dist_id] != 0 ) { 
					//			x_dxprime = ( image_point[0].x - image_point_perturbed[0].x ); 
					//			y_dyprime = ( image_point[0].y - image_point_perturbed[0].y ); 
					//		}

					//		x_u2 += x_dxprime * x_dxprime;
					//		y_u2 += y_dyprime * y_dyprime;

					//	}*/

					//	//x_u2 += cameras[cam_id].centriod_loc_uncertainty * cameras[cam_id].centriod_loc_uncertainty; // 0.25 = 0.5 * 0.5 = squared uncertainty of centriod pixel coordinate
					//	//y_u2 += cameras[cam_id].centriod_loc_uncertainty * cameras[cam_id].centriod_loc_uncertainty; // 0.25 = 0.5 * 0.5 = squared uncertainty of centriod pixel coordinate

					//	/*double x = ( P->x - cameras[cam_id].c[0] ) / cameras[cam_id].f[0];
					//	double y = ( P->y - cameras[cam_id].c[1] ) / cameras[cam_id].f[1];

					//	double dximdx = 1.0 / cameras[cam_id].f[0];
					//	double dximdc = -1.0 / cameras[cam_id].f[0];
					//	double dximdf = ( cameras[cam_id].c[0] - P->x ) / ( cameras[cam_id].f[0] * cameras[cam_id].f[0] );

					//	cameras[cam_id].X_u[0] = 
					//		sqrt( 
					//		dximdx * dximdx * x_u2
					//		+ dximdc * dximdc * cameras[cam_id].c_u[0] * cameras[cam_id].c_u[0]
					//		+ dximdf * dximdf * cameras[cam_id].f_u[0] * cameras[cam_id].f_u[0]
					//	);

					//	double dyimdy =  1.0 / cameras[cam_id].f[1];
					//	double dyimdc = -1.0 / cameras[cam_id].f[1];
					//	double dyimdf = ( cameras[cam_id].c[1] - P->y ) / ( cameras[cam_id].f[1] * cameras[cam_id].f[1] );

					//	cameras[cam_id].X_u[1] = 
					//		sqrt( 
					//		dyimdy * dyimdy * y_u2 
					//		+ dyimdc * dyimdc * cameras[cam_id].c_u[1] * cameras[cam_id].c_u[1]
					//		+ dyimdf * dyimdf * cameras[cam_id].f_u[1] * cameras[cam_id].f_u[1]
					//		);*/

					//	//// 3D position uncertainty due to x' and y' image coordinates
					//	/*if ( cameras[cam_id].X_u[0] != 0 && cameras[cam_id].X_u[1] != 0) {
					//		for (int dim = 0; dim < 2; ++dim) {

					//			cv::Mat delta_B = cv::Mat::zeros( 2 * cameras.size(), 1, CV_64F );
					//			cv::Mat E = cv::Mat::zeros( 2 * cameras.size(), 3, CV_64F );

					//			int s = (dim == 0 ? cam_id * 2 : cam_id * 2 + 1);

					//			E.at<double>(s, 0) = cameras[cam_id].X_u[dim] * cameras[cam_id].R[2][0];   
					//			E.at<double>(s, 1) = cameras[cam_id].X_u[dim] * cameras[cam_id].R[2][1];
					//			E.at<double>(s, 2) = cameras[cam_id].X_u[dim] * cameras[cam_id].R[2][2];
					//			delta_B.at<double>(s) = -1.0 * cameras[cam_id].X_u[dim] * cameras[cam_id].T[2];

					//			svd(E);
					//			double max_s = 0, min_s = 0; 
					//			cv::minMaxIdx(svd.w, &min_s, &max_s);
					//			double normL2_E = max_s;
					//			double normL2_deltaB = cv::norm(delta_B);
					//			double E_sen = (cond_A * cond_A * tan(theta) + cond_A ) * normL2_E / normL2_A * normL2_X / cameras[cam_id].X_u[dim];
					//			double deltaB_sen = cond_A / cos(theta) * normL2_deltaB / normL2_B * normL2_X / cameras[cam_id].X_u[dim];
					//			double squared_uncertainty_homogenious_coordinate =	
					//				( E_sen * E_sen + deltaB_sen * deltaB_sen) * cameras[cam_id].X_u[dim] * cameras[cam_id].X_u[dim];

					//			newparticle->uncertainty += squared_uncertainty_homogenious_coordinate;
					//		}
					//	}*/
					//	//// 3D position uncertainty to translation vector T
					//	//for (int t = 0; t < 3; ++t) {
					//	//	double T_u[3] = {0};
					//	//	T_u[t] = cameras[cam_id].T_u[t];

					//	//	cv::Mat delta_B = cv::Mat::zeros( 2 * cameras.size(), 1, CV_64F );

					//	//	int s = cam_id * 2;
					//	//	int e = s + 1;

					//	//	delta_B.at<double>(s) = T_u[0] - x * T_u[2];
					//	//	delta_B.at<double>(e) = T_u[1] - y * T_u[2];

					//	//	double normL2_deltaB = cv::norm(delta_B);
					//	//	double deltaB_sen = 0;
					//	//	if ( T_u[t] != 0)
					//	//		deltaB_sen = cond_A / cos(theta) * normL2_deltaB / normL2_B * normL2_X / cameras[cam_id].T_u[t];
					//	//	double squared_uncertainty_tvec =	deltaB_sen * deltaB_sen * T_u[t] * T_u[t];

					//	//	newparticle->uncertainty += squared_uncertainty_tvec;		
					//	//}

					//	//// 3D position uncertainty due to Rotation vector r_vec


					//	//cv::Mat A_u = cv::Mat::zeros( 2 * cameras.size(), 3, CV_64F );
					//	//cv::Mat B_u = cv::Mat::zeros( 2 * cameras.size(), 1, CV_64F );
					//	//cv::Mat X_u = cv::Mat::zeros( 3, 1, CV_64F );
					//	////cout << "\n" << i << "\t";
					//	//for(int c = 0; c < cameras.size(); ++c) {
					//	//	lpt::ParticleImage* Pc = matches[m]->particles[c].first;
					//	//	int cam_id = matches[m]->particles[c].second;

					//	//	double xc = ( Pc->x - cameras[c].c[0] ) / cameras[c].f[0];
					//	//	double yc = ( Pc->y - cameras[c].c[1] ) / cameras[c].f[1];

					//	//	cv::Mat R_u = cv::Mat(3, 3, CV_64F, cameras[c].R).clone(); 		
					//	//	cv::Mat r_vec(3, 1, CV_64F);
					//	//	cv::Rodrigues(R_u, r_vec);

					//	//	r_vec.at<double>(0) += cameras[c].r_vec_u[0];
					//	//	r_vec.at<double>(1) += cameras[c].r_vec_u[1];
					//	//	r_vec.at<double>(2) += cameras[c].r_vec_u[2];

					//	//	cv::Rodrigues(r_vec, R_u);

					//	//	int s = c * 2;
					//	//	int e = s + 1;

					//	//	A_u.at<double>(s, 0) = xc * R_u.at<double>(2,0) - R_u.at<double>(0,0);   
					//	//	A_u.at<double>(s, 1) = xc * R_u.at<double>(2,1) - R_u.at<double>(0,1);
					//	//	A_u.at<double>(s, 2) = xc * R_u.at<double>(2,2) - R_u.at<double>(0,2);

					//	//	A_u.at<double>(e, 0) = yc * R_u.at<double>(2,0) - R_u.at<double>(1,0);
					//	//	A_u.at<double>(e, 1) = yc * R_u.at<double>(2,1) - R_u.at<double>(1,1);
					//	//	A_u.at<double>(e, 2) = yc * R_u.at<double>(2,2) - R_u.at<double>(1,2);

					//	//	B_u.at<double>(s) = cameras[c].T[0] - xc * cameras[c].T[2];
					//	//	B_u.at<double>(e) = cameras[c].T[1] - yc * cameras[c].T[2];
					//	//}

					//	//svd(A_u);
					//	//svd.backSubst(B_u, X_u);
					//	//double dX = 
					//	//	sqrt(
					//	//	(newparticle->X[0] - X_u.at<double>(0)) * (newparticle->X[0] - X_u.at<double>(0)) +
					//	//	(newparticle->X[1] - X_u.at<double>(1)) * (newparticle->X[1] - X_u.at<double>(1)) +
					//	//	(newparticle->X[2] - X_u.at<double>(2)) * (newparticle->X[2] - X_u.at<double>(2)) 
					//	//	);

					//	//double squared_uncertainty_rvec = dX*dX;

					//	//newparticle->uncertainty += squared_uncertainty_rvec;					

					//}
					//newparticle->uncertainty = ( newparticle->uncertainty > 0 ? sqrt(newparticle->uncertainty) : 0 );

					frame.objects.push_back(std::move(newparticle));
				}
			}
		}
	}


	////------OLD CODE---------------------------------------------------------------------------
	//void SpatialMatcher::findMatches() {
	//	cout << "Starting correspondence search: Total Frames = " << (int)cameras[0].frames.size() <<  endl;
	//    globalmatches.resize( (int)cameras[0].frames.size() );
	//    for (int frame_index = 0; frame_index < (int)cameras[0].frames.size(); ++frame_index) {
	////    	cout << "Frame " << frame_index << ":" << endl;
	////    	cout << "\t Finding two way matches: ";
	//    	for (vector<CameraPair>::iterator pair = stereo_pairs.begin(); pair != stereo_pairs.end(); ++pair){
	//        	find2WayMatches(frame_index, *pair);
	//        }
	////        cout<<"\t--Complete" <<endl << "\t Finding multicam matches:";
	//
	//        for (int camA = 0; camA < cameras.size() - 1; ++camA) {
	//            // printf("%d,", cameras_[camA]->ID_);
	//        	bool matched4way = false;
	//            for (int i = 0; i < cameras[camA].frames[frame_index].particles.size(); ++i) {
	//                ParticleImage* Pa = cameras[camA].frames[frame_index].particles[i].get();
	//                for (int camB = camA + 1; camB < cameras.size() - 1; ++camB) {
	//                    // printf("%d,", cameras_[camB]->ID_);
	//                    set<ParticleImage*>& matchesAB = Pa->matches2way[camB];
	//                    set<ParticleImage*>::iterator matchAB;
	//                    for(matchAB = matchesAB.begin(); matchAB != matchesAB.end(); ++matchAB) { //loop through all A,B matches
	//                        ParticleImage* Pb = (*matchAB);
	//
	//                        for (int camC = camB + 1; camC < cameras.size(); ++camC) {
	//                            // printf("%d\n", cameras_[camC]->ID_);
	//                            set<ParticleImage*> &matchesBC = Pb->matches2way[camC];
	//                            set<ParticleImage*>::iterator matchBC;
	//
	//                            for (matchBC = matchesBC.begin(); matchBC != matchesBC.end(); ++matchBC) {
	//                                ParticleImage* Pc = (*matchBC);
	//                                for (int camD = camC + 1; camD < cameras.size(); ++camD) {
	//                                    // printf("%d\n", cameras_[camD]->ID_);
	//                                    set<ParticleImage*> &matchesCD = Pc->matches2way[camD];
	//                                    set<ParticleImage*>::iterator matchCD;
	//                                    for (matchCD = matchesCD.begin(); matchCD != matchesCD.end(); ++matchCD) {
	//                                        ParticleImage* Pd = (*matchCD);
	//                                        if(Pd->matches2way[camA].count(Pa) && Pb->matches2way[camD].count(Pd) && Pa->matches2way[camC].count(Pc)) {
	//                                           //printf("\t SuperMATCH!!!!");
	//                                            Match* newmatch = new Match();
	//                                            newmatch->addParticle(Pa,camA);
	//                                            newmatch->addParticle(Pb,camB);
	//                                            newmatch->addParticle(Pc,camC);
	//                                            newmatch->addParticle(Pd,camD);
	//                                            globalmatches[frame_index].push_back(newmatch);
	////                                            printf("4Matching: %d-%d, %d-%d, %d-%d, %d-%d\n", camA, Pa->id, camB, Pb->id, camC, Pc->id, camD, Pd->id);
	//                                            matched4way = true;
	//                                        } //TODO: Add code to identify simple 4-way matches 1234, 1243 and 1324.
	//                                    }
	//                                }
	//                                //TODO: add code to store the match residual in the Match object
	//                                if (matched4way == false)
	//
	//                                    if( Pc->matches2way[camA].count(Pa) ) {
	//                                    	;
	////                                        Match* newmatch = new Match();
	////                                        newmatch->addParticle(Pa,camA);
	////                                        newmatch->addParticle(Pb,camB);
	////                                        newmatch->addParticle(Pc,camC);
	////                                        globalmatches[frame_index].push_back(newmatch);
	////                                        printf("3Matching: %d-%d, %d-%d, %d-%d\n", camA, Pa->id, camB, Pb->id, camC, Pc->id);
	//                                    }
	//                            }
	//                        }
	//                    }
	//                }
	//            }
	//        }
	////        cout<< "\t--Complete"<< endl << "\t Removing non unique matches";
	//        removeNonUniqueMatches( globalmatches[frame_index] );
	////        cout<< "\t--Complete"<<endl<<endl;
	//    }
	//}
	//
	///*void SpatialMatcher::InitializeMatchMap(int camID, ParticleImage* P) {
	//
	//    for (int c = 0; c < number_of_cameras_; ++c) {
	//        matchmaps_[c];
	//    }
	//}*/
	//
	//void SpatialMatcher::find2WayMatches(int frame_index, CameraPair &pair) {
	//    vector<ParticleImage::Ptr>& cam_A_particles = pair.cam_A.frames[frame_index].particles;
	//    vector<ParticleImage::Ptr>& cam_B_particles = pair.cam_B.frames[frame_index].particles;
	//
	//    for (int b = 0; b < (int)cam_B_particles.size(); ++b) {
	//        double lineA[3];
	//        calculateEpiline(cam_B_particles[b], pair.F, lineA);
	//        for (int a = 0; a < (int)cam_A_particles.size(); ++a) {
	//            double residual = calculateEpipolarResidual(lineA, cam_A_particles[a]);
	////            if (i == j){
	////                printf("Should match C%d_%d and C%d_%d, r = %f\n", pair.cam_A_id, cam_A_particles[a]->id, pair.cam_B_id, cam_B_particles[b]->id, residual);
	////                //printf("Cam%d_%d: [%E, %E]\n", A_->ID_, Cam_A_particles[a]->id_,Cam_A_particles[a]->px_, Cam_A_particles[a]->py_);
	////                //printf("Cam%d_%d: [%E, %E]\n", B_->ID_, Cam_B_particles[b]->id_,Cam_B_particles[b]->px_, Cam_B_particles[b]->py_);
	////            }
	//            if (residual <= match_residual_threshold){
	//                cam_A_particles[a]->add2WayMatch(pair.cam_B.id, cam_B_particles[b].get());
	//                cam_B_particles[b]->add2WayMatch(pair.cam_A.id, cam_A_particles[a].get());
	//                //matches_[ Cam_A_particles[i] ].insert( Cam_B_particles[j] );
	//                //FIXME: Find a way to store the residual value and later add 3 and 4 cam matches
	////                printf("matched C%d_%d and C%d_%d, r = %f\n", pair.cam_A_id, cam_A_particles[i]->id, pair.cam_B_id, cam_B_particles[j]->id, residual);
	//            }
	//        }
	//    }
	//
	//}
	//
	//double SpatialMatcher::calculateEpipolarResidual(double line[3], ParticleImage::Ptr point)  {
	//    double residual = fabs( point->x * line[0] + point->y * line[1] + line[2] );
	//    return residual;
	//}
	//
	//void SpatialMatcher::calculateEpiline(ParticleImage::Ptr point, const double F[3][3], double line[3]) {
	//    double x = point->x;
	//    double y = point->y;
	//    double z = 1.0;
	//
	//    line[0] = F[0][0] * x + F[1][0] * y + F[2][0] * z;
	//    line[1] = F[0][1] * x + F[1][1] * y + F[2][1] * z;
	//    line[2] = F[0][2] * x + F[1][2] * y + F[2][2] * z;
	//    double factor = line[0] * line[0] + line[1] * line[1];
	//    factor = factor ? 1./sqrt(factor) : 1.;
	//    line[0] *= factor;
	//    line[1] *= factor;
	//    line[2] *= factor;
	//}
	//
	//void SpatialMatcher::removeNonUniqueMatches(vector<Match*> &matches) {
	//    for (int i = 0; i < matches.size(); ++i)
	//        if (matches[i]->isUnique() == false) {
	//            delete matches[i];//printf(" not unique\n");
	//            matches[i] = 0;
	//            // TODO: Erase null matches
	//        }
	//}
	//
	//void SpatialMatcher::get3wayMatch(int camB, int camC, ParticleImage::Ptr Pa) {
	//    set<ParticleImage*>::iterator matchAB;
	//    for (matchAB = Pa->matches2way[camB].begin(); matchAB != Pa->matches2way[camB].end(); ++matchAB){
	//        ParticleImage* Pb = (*matchAB);
	//        set<ParticleImage*> &matchesBC = Pb->matches2way[camC];
	//        set<ParticleImage*>::iterator matchAC;
	//        for (matchAC = Pa->matches2way[camC].begin(); matchAC != Pa->matches2way[camC].end(); ++matchAC) {
	//            ParticleImage* particle_matched_AC = (*matchAC);
	//            if(matchesBC.count(particle_matched_AC)) //FIXME: this compares two pointers and my error if parallelized
	//                printf("matched: %d,%d,%d \n", Pa->id, Pb->id, particle_matched_AC->id);
	//        }
	//    }
	//}
	//
	//void SpatialMatcher::testMatches(){
	//	for (int f = 0; f < globalmatches.size(); f++) {
	//	    	int N_correct = 0;
	//	    	int N_matched = 0;
	//	    	int N_total = 100000000;
	//	    	for(int c = 0; c < cameras.size(); c++) {
	//	    			int temp = cameras[c].frames[f].particles.size();
	//	    			if (temp < N_total)
	//	    				N_total = temp;
	//	    	}
	//
	//	    	for (int m = 0; m < globalmatches[f].size(); m++){
	//	    		if ( globalmatches[f][m] ) {
	//	    			N_matched++;
	//	    			vector<pair < ParticleImage*, int > >& particles = globalmatches[f][m]->particles;
	//	    			bool matched = true;
	//	    			for (int j = 0; j < particles.size() - 1; j++){
	//	    				matched &= particles[j].first->id == particles[j+1].first->id ? true : false;
	//	    			}
	//	    			if ( matched )
	//	    				N_correct++;
	//	    		}
	//	    	}
	//	    	cout << "Frame "<< f << ":";
	//	        cout << "\t Correct Ratio: " << N_correct << " / " << N_matched << " = " <<
	//	        		setprecision(2) << (N_matched > 0 ? (double) N_correct / N_matched : 0.0 );
	//	        cout << "\t Cover Ratio: " << N_correct << " / " << N_total << " = " <<
	//	        		setprecision(2) <<  (N_total > 0 ? (double) N_correct / N_total : 0.0 ) << endl;
	//	    }
	//}
	//
	//SpatialMatcher::~SpatialMatcher() {
	//  //TODO: Auto generated destructor
	//}
	//
	//void reconstruct3D(vector<lpt::Camera>& cameras, vector<vector<lpt::Match*> >& matches, vector<lpt::Frame*>& frames) {
	//	lpt::Regression solve;
	//	for (int f = 0; f < matches.size(); ++f) {
	//		cout << "Frame " << f <<":  number of matches = " << matches[f].size() << endl;
	//		lpt::Frame* newframe = new lpt::Frame(f);     // FIXME: Assumes frame_index is sequential starting at zero!!!
	//		for (int m = 0; m < matches[f].size(); ++m) {
	//			if (matches[f][m]) { //if valid pointer
	//				int number_of_cams = matches[f][m]->particles.size();
	//				//cout << "\t match " << m << ":  number of cameras = " << number_of_cams <<  endl;
	//				double empty[3] = {0};
	//				double empty1 = 0;
	//				double** A = new double* [ 2 * number_of_cams ];
	//				double* B = new double [ 2 * number_of_cams ];
	//
	//				double* X = new double[3];
	//				int id = -1;
	//				for(int i = 0; i < number_of_cams; ++i) {
	//					int s = i * 2;
	//					int e = s + 1;
	//					A[s] = new double[3];
	//					A[e] = new double[3];
	//					lpt::ParticleImage* P = matches[f][m]->particles[i].first;
	//					int camID = matches[f][m]->particles[i].second;
	//					double x = ( P->x - cameras[camID].c[0] ) / (1. * cameras[camID].f[0]);    // FIXME: May need -1 multiplier for pixel coordinate system (upper left corner origin)
	//					double y = ( P->y - cameras[camID].c[1] ) / (1. * cameras[camID].f[1]);
	//					id = P->id;
	//					A[s][0] = x * cameras[camID].R[2][0] - cameras[camID].R[0][0];   // Convert P.x and P.y to normalized coordinates through intrinsic parameters
	//					A[s][1] = x * cameras[camID].R[2][1] - cameras[camID].R[0][1];
	//					A[s][2] = x * cameras[camID].R[2][2] - cameras[camID].R[0][2];
	//					A[e][0] = y * cameras[camID].R[2][0] - cameras[camID].R[1][0];
	//					A[e][1] = y * cameras[camID].R[2][1] - cameras[camID].R[1][1];
	//					A[e][2] = y * cameras[camID].R[2][2] - cameras[camID].R[1][2];
	//
	//					B[s] = cameras[camID].T[0] - x * cameras[camID].T[2];
	//					B[e] = cameras[camID].T[1] - y * cameras[camID].T[2];
	//				}
	//
	//				solve.Householder(A, B, number_of_cams * 2, 3);   //Transform A into upper triangular form
	////				cout << "ready to solve: " << m << endl;
	////				for (int a = 0; a < number_of_cams * 2; a++){
	////					for (int b = 0; b < 3; b++){
	////						cout << A[a][b] << " ";
	////					}
	////					cout << endl;
	////				}
	////				cout << endl << "B = " << endl;
	////				for (int a = 0; a < number_of_cams * 2; a++)
	////					cout << B[a] << endl;
	//				solve.Backsub(A, B, 3, X);
	//				//cout << "X = " << X[0] << ", "<< X[1] << ", " << X[2]<< endl;
	//
	//				lpt::Particle::Ptr  newparticle = lpt::Particle::create(id, X[0], X[1], X[2]);
	//				newframe->particles.push_back(newparticle);
	//				for(int i = 0; i < number_of_cams * 2; ++i)
	//					delete [] A[i];
	//				delete [] A;
	//				delete [] B;
	//				delete [] X;
	//			}
	//		}
	//		frames.push_back(newframe);
	//	}
	//}

} /* namespace lpt */
