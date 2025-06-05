
#include "datagen.hpp"

namespace lpt {

using namespace std;

void convertFrame(const vector<cv::Point2f>& image_points, lpt::ImageFrame& frame, vector<int>& particleIDs) {
	for (int j = 0; j < image_points.size(); ++j) {
		lpt::ParticleImage::Ptr newparticle = lpt::ParticleImage::create(particleIDs[j], image_points[j].x, image_points[j].y);
		frame.particles.push_back(newparticle);
	}
}

void convertCamParameters2CV(const lpt::Camera& cam, cv::Mat& camera_matrix, cv::Mat& dist_coeffs,
		cv::Mat& rotation_matrix, cv::Mat& translation_vec)
{
	camera_matrix =  cv::Mat::eye(3, 3, CV_64F);
	rotation_matrix = cv::Mat::zeros(3, 3, CV_64F);
	translation_vec = cv::Mat::zeros(3, 1, CV_64F);
	dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

	camera_matrix.at<double>(0,0) = cam.f[0];
	camera_matrix.at<double>(1,1) = cam.f[1];
	camera_matrix.at<double>(0,2) = cam.c[0];
	camera_matrix.at<double>(1,2) = cam.c[1];

	dist_coeffs.at<double>(0) = cam.dist_coeffs[0];
	dist_coeffs.at<double>(1) = cam.dist_coeffs[1];
	dist_coeffs.at<double>(2) = cam.dist_coeffs[2];
	dist_coeffs.at<double>(3) = cam.dist_coeffs[3];

	for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				rotation_matrix.at<double>(i,j) = cam.R[i][j];

	translation_vec.at<double>(0) = cam.T[0];
	translation_vec.at<double>(1) =	cam.T[1];
	translation_vec.at<double>(2) =	cam.T[2];
}

void setCameraRotation(lpt::Camera& cam, double angle[3]) {
	double alpha = angle[0];
	double beta = angle[1];
	double gamma = angle[2];

	double rx[] =
	        {
	            1.0,	0.0,	0.0,
	            0.0,	cos(alpha), -sin(alpha),
	            0.0,	sin(alpha), cos(alpha)
	        };
	cv::Mat Rx(3, 3, CV_64F, rx);

	double ry[] =
	        {
	            cos(beta),	0.0,	sin(beta),
	            0.0,	1.0, 	0.0,
	            -sin(beta),	0.0, cos(beta)
	        };
	cv::Mat Ry(3, 3, CV_64F, ry);

	double rz[] =
	        {
	            cos(gamma),	-sin(gamma),	0.0,
	            sin(gamma),	cos(gamma),		0.0,
	            0.0,	0.0, 	1.0
	        };
	cv::Mat Rz(3, 3, CV_64F, rz);
	cv::Mat Rmat = Rx * Ry * Rz;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			cam.R[i][j] = Rmat.at<double>(i,j);
}

void setCameraTranslation(lpt::Camera& cam, double trans[3]) {
	for (int i = 0; i < 3; ++i)
		cam.T[i] = trans[i];
}

void setCameraIntrinsics(lpt::Camera& cam,
		double focal_length, double pixel_width,
		double aspect_ratio, int image_width,
		int image_height, double dist_coeffs[4])
{
	cam.f[0] = focal_length / pixel_width;
	cam.f[1] = focal_length / (pixel_width * aspect_ratio );
	cam.c[0] = (double)image_width / 2.0 - 0.5;
	cam.c[1] = (double)image_height / 2.0 - 0.5;
	cam.dist_coeffs[0] = dist_coeffs[0];
	cam.dist_coeffs[1] = dist_coeffs[1];
	cam.dist_coeffs[2] = dist_coeffs[2];
	cam.dist_coeffs[3] = dist_coeffs[3];
}

void calcFundamentalMatrices( vector<lpt::CameraPair>& camera_pairs) {
	
	for (vector<lpt::CameraPair>::iterator pair_it = camera_pairs.begin();
			pair_it != camera_pairs.end(); ++pair_it)
	{
		int cam_a_id = pair_it->cam_A.id;
		int cam_b_id = pair_it->cam_B.id;
		cout << "Calculating F for " << cam_a_id << " and " << cam_b_id << endl;
		cv::Mat Ma, Mb;
		cv::Mat Ra, Rb, Ta, Tb;
		cv::Mat notused;
		lpt::convertCamParameters2CV(pair_it->cam_A, Ma, notused, Ra, Ta);
		lpt::convertCamParameters2CV(pair_it->cam_B, Mb, notused, Rb, Tb);

		cv::Mat R = cv::Mat::zeros(3,3,CV_64F);
		cv::Mat T = cv::Mat::zeros(3,1,CV_64F);
		cv::Mat E = cv::Mat::zeros(3,3,CV_64F);
		cv::Mat F = cv::Mat::zeros(3,3,CV_64F);
		cv::Mat Fs;

		R = Rb * Ra.t();
		T = Ta - (R.t() * Tb);

		double s[] = {
					 0.0, 	-T.at<double>(2),	T.at<double>(1),
					 T.at<double>(2), 	0.0,   -T.at<double>(0),
					-T.at<double>(1), 	T.at<double>(0), 	0.0
					};
		cv::Mat S(3, 3, CV_64F, s);

		E = R * S;
		F = Mb.inv().t() * E * Ma.inv();

		double scale = fabs( F.at<double>(2,2) ) > 0 ? 1.0 / F.at<double>(2,2) : 1.0;
		F.convertTo(Fs, CV_64F, scale);

		for ( int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				pair_it->F[i][j] = Fs.at<double>(i,j);

//		for (int c = 0; c < cameras[cam_a].frames[0].particles.size(); c++) {
//			cv::Mat pa (3, 1, CV_64F);
//			pa.at<double>(0) = cameras[cam_a].frames[0].particles[c]->x;
//			pa.at<double>(1) = cameras[cam_a].frames[0].particles[c]->y;
//			pa.at<double>(2) = 1.0;
//
//			cv::Mat pb (3, 1, CV_64F);
//			pb.at<double>(0) = cameras[cam_b].frames[0].particles[c]->x;
//			pb.at<double>(1) = cameras[cam_b].frames[0].particles[c]->y;
//			pb.at<double>(2) = 1.0;
//
//			cout << "resid ab = " << c << "  " << pa.t() * F * pb << endl;
//			cout << "resid ba = " << pb.t() * F * pa << endl;
//		}
	}
}

double distance(const cv::Point2f& a, const cv::Point2f& b) {
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double distance(double x, double y, double x_center, double y_center) {
	return sqrt((x_center - x) * (x_center - x) + (y_center - y) * (y_center - y));
}

double gaussian_2d(double x, double y, double x_center, double y_center, double sigma) {
	double temp_distance = sqrt((x_center - x) * (x_center - x) + (y_center - y) * (y_center - y));
	return exp(-temp_distance * temp_distance / (2.0 * sigma * sigma));
}

void draw_blur_line(cv::Mat& img, const cv::Point2f& a, const cv::Point2f& b, double sigma, int radius) {
	double max_intensity = 100.0;

	double length = distance(a, b);
	cv::Point2f direction = (b - a) / length;

	int cols = img.cols;
	int rows = img.rows;

	int n_steps = 100;
	double step = length / (double)(n_steps-1);

	//cout << "a: " <<". x: " << a.x << ". y: " << a.y << endl;
	//cout << "b: " <<". x: " << b.x << ". y: " << b.y << endl;
	//cout << "length is: " << length << endl;
	//cout << "step is: " << step << endl;
	//cout << "direction is: " << direction.x << " " << direction.y << endl;
	//cout << "end: " << ". x: " << (a + direction * (n_steps-1) * step).x << ". y: " << (a + direction * (n_steps - 1) * step).y << endl;

	if (length < 1.5) {
		int x = (int)a.x;
		int y = (int)a.y;

		if (x - radius >= 0 && x + radius < cols && y - radius >= 0 && y + radius < rows) {
			for (int m = -radius; m <= radius; m++) {
				for (int n = -radius; n <= radius; n++) {
					double xx = (double)(m + x) + 0.5;
					double yy = (double)(n + y) + 0.5;
					double x_center = a.x;
					double y_center = a.y;

					double intensity = max_intensity * gaussian_2d(xx, yy, x_center, y_center, sigma);

					img.at<uchar>(n + y, m + x) = (uchar)intensity;
				}
			}
		}
	}
	else {
		double xx, yy, x_center, y_center, intensity;
		double weight;
		cv::Mat count_img = cv::Mat::zeros(rows, cols, CV_8UC1);
		cv::Mat intensity_img = cv::Mat::zeros(rows, cols, CV_32FC1);
		cv::Mat weight_img = cv::Mat::zeros(rows, cols, CV_32FC1);

		for (int s = 0; s < n_steps; s++) {
			cv::Point2f p = a + direction * step * s;

			int x = (int)p.x;
			int y = (int)p.y;

			//cout << "s: " << s << ". x: " << x << ". y: " << y << endl;

			if (x - radius >= 0 && x + radius < cols && y - radius >= 0 && y + radius < rows) {
				for (int m = -radius; m <= radius; m++) {
					for (int n = -radius; n <= radius; n++) {
						xx = (double)(m + x) + 0.5;
						yy = (double)(n + y) + 0.5;
						x_center = p.x;
						y_center = p.y;

						intensity = gaussian_2d(xx, yy, x_center, y_center, sigma);
						weight = 1.0 / (distance(xx, yy, x_center, y_center) * distance(xx, yy, x_center, y_center));
						//weight = 1.0 / intensity;

						intensity_img.at<float>(n + y, m + x) = intensity_img.at<float>(n + y, m + x) + (float)(intensity * weight);
						weight_img.at<float>(n + y, m + x) = weight_img.at<float>(n + y, m + x) + (float)weight;
						//count_img.at<uchar>(n + y, m + x) = count_img.at<uchar>(n + y, m + x) + 1;
					}
				}
			}

			// average the intensity
			if (x - radius >= 0 && x + radius < cols && y - radius >= 0 && y + radius < rows) {
				for (int m = -radius; m <= radius; m++) {
					for (int n = -radius; n <= radius; n++) {
						img.at<uchar>(n + y, m + x) = (uchar)(max_intensity * (double)intensity_img.at<float>(n + y, m + x) / (double)weight_img.at<float>(n + y, m + x));

						//img.at<uchar>(n + y, m + x) = (uchar)(max_intensity * (double)intensity_img.at<float>(n + y, m + x) / (double)count_img.at<uchar>(n + y, m + x));
					}
				}
			}
		}
	}

	// draw the true point as black point
	//cout << (int)img.at<uchar>((int)a.y, (int)a.x) << endl;

}

void draw_gaussian_blob(cv::Mat& img, const cv::Point2f& p, double sigma, int radius) {
	double max_intensity = 255.0;


	int cols = img.cols;
	int rows = img.rows;

	int x = (int)p.x;
	int y = (int)p.y;

	if (x - radius >= 0 && x + radius < cols && y - radius >= 0 && y + radius < rows) {
		for (int m = -radius; m <= radius; m++) {
			for (int n = -radius; n <= radius; n++) {
				double xx = (double)(m + x) + 0.5;
				double yy = (double)(n + y) + 0.5;
				double x_center = p.x;
				double y_center = p.y;

				double intensity = max_intensity * gaussian_2d(xx, yy, x_center, y_center, sigma);

				img.at<uchar>(n + y, m + x) = (uchar)intensity;
			}
		}

	}
}

void ImageCreator::createGaussianImage(lpt::ImageFrame& frame) {
	frame.image = image_type.clone();
	int height = image_type.rows;
	int width = image_type.cols;

	for (int p = 0; p < frame.particles.size(); ++p) {
		double r, I;
		r = 2;

		cv::Point2f point(frame.particles[p]->x, frame.particles[p]->y);
		draw_gaussian_blob(frame.image, point, 1.0, r);
	}
}

void ImageCreator::createBlurStreamsImage(lpt::ImageFrame& frame) {
	frame.image = image_type.clone();
	int height = image_type.rows;
	int width = image_type.cols;

	for (int p = 0; p < frame.streaks_floats.size(); ++p) {
		double r, I;
		r = 2;
		I = 255;

		vector<cv::Point2f> v = frame.streaks_floats[p]->curvePoints2f;

		// v[0] is the start point
		draw_blur_line(frame.image, v[0], v[2], 1, 1);

		//cv::Point2f line_0 = v[0] - v[1];
		//cv::Point2f line_1 = v[1] - v[2];
		//cout << "line0: " << line_0.x << "; " << line_0.y << endl;
		//cout << "line1: " << line_1.x << "; " << line_1.y << endl;

	}
}

void ImageCreator::createStreaksImage(lpt::ImageFrame& frame) {
	frame.image = image_type.clone();
	int height = image_type.rows;
	int width = image_type.cols;

	for (int p = 0; p < frame.streaks.size(); ++p) {
		double x = frame.particles[p]->x;
		double y = frame.particles[p]->y;
		double r, I;

		if (this->radius > 0)
			r = this->radius;
		else
			r = frame.particles[p]->radius;

		if (this->intensity > 0)
			I = this->intensity;
		else
			I = frame.particles[p]->intensity;
		if (r >= 1) {
			cv::polylines(frame.image, frame.streaks[p]->curvePoints, false, I, r);
		}
		else {
			double temp_x = 0.0;
			double temp_y = 0.0;
			for (int i = 0; i < frame.streaks[p]->curvePoints.size(); i++) {
				temp_x = frame.streaks[p]->curvePoints[i].x;
				temp_y = frame.streaks[p]->curvePoints[i].y;
				if (static_cast<int>(temp_y) >= 0 && static_cast<int>(temp_y) < height && static_cast<int>(temp_x) >= 0 && static_cast<int>(temp_x) < width)
					frame.image.at<uchar>(static_cast<int>(temp_y), static_cast<int>(temp_x)) = static_cast<unsigned char>(I);
			}
		}
	}
	cv::GaussianBlur(frame.image, frame.image, cv::Size(blur_ksize, blur_ksize), 0, 0);
}

void ImageCreator::createImage(lpt::ImageFrame& frame) {
	frame.image = image_type.clone();

	int height = image_type.rows;
	int width = image_type.cols;

	for ( int p = 0; p < frame.particles.size(); ++p) {
		double x = frame.particles[p]->x;
		double y = frame.particles[p]->y;
		double r, I;
		
		if (this->radius > 0)
			r = this->radius;
		else
			r = frame.particles[p]->radius;

		if (this->intensity > 0)
			I = this->intensity;
		else
			I = frame.particles[p]->intensity;		
		if ( r >= 1 ) {
			cv::Point center(static_cast<int>(x), static_cast<int>(y));
			cv::circle(frame.image, center, static_cast<int>(r), cv::Scalar( I, I, I ), -1, 8 );
		} else {
			if(static_cast<int>(y) >=0 && static_cast<int>(y) < height && static_cast<int>(x) >=0 && static_cast<int>(x) < width)
				frame.image.at<uchar>(static_cast<int>(y), static_cast<int>(x)) = static_cast<unsigned char>(I);
		}
	}
	cv::GaussianBlur( frame.image, frame.image, cv::Size( blur_ksize, blur_ksize ), 0, 0 );
}

void DataSetGenerator::polyFitTraj(Trajectory& traj, int degree, double interval) {
	int m = traj.particles.size();

	if (m <= 2)
		return;

	vector<double> time(m);
	vector<double> x(m);
	vector<double> y(m); 
	vector<double> z(m);

	vector<double> fitX(m);

	for (int i = 0; i < m; i++) {
		time[i] = (i) * interval;
		x[i] = traj.particles[i]->x;
		y[i] = traj.particles[i]->y;
		z[i] = traj.particles[i]->z;
	}

	vector<double> params_x(degree);
	vector<double> params_y(degree);
	vector<double> params_z(degree);

	solver.Polyfit(time.data(), x.data(), m, degree, params_x.data());

	//for (int i = 0; i < m; i++) {
	//	fitX[i] = params_x[0] + params_x[1] * time[i] + params_x[2] * time[i] * time[i];
	//	cout << "original x is: " << x[i] << ". Fitted x is: " << fitX[i] << endl;


	//}
	//for (int i = 0; i < m; i++) {
	//	if (i > 0 && i < m - 1) {
	//		double diff_v = (x[i + 1] - x[i - 1]) / (2 * interval);
	//		double diff_vd = (fitX[i + 1] - fitX[i - 1]) / (2 * interval);
	//		double diff_d = 2 * params_x[2] * time[i] + params_x[1];

	//		cout << "diff_v is: " << diff_v << ". diff_d is: " << diff_d << ". diff_vd is: " << diff_vd << endl;
	//	}
	//}


}

vector<lpt::Trajectory::Ptr> DataSetGenerator::generateAdditionalTrajectories(int timesOfTrajs, vector<lpt::Trajectory::Ptr>& trajs) {
	vector<lpt::Trajectory::Ptr> tempTrajs;
	double tempX, tempY, tempZ;
	double spacing = 66.0;
	if (timesOfTrajs > 0) {
		for (int j = 0; j < trajs.size(); j++) {
			for (int i = 0; i < timesOfTrajs; i++) {
				tempX = ((double)rand() / RAND_MAX - 0.5) * spacing;
				tempY = ((double)rand() / RAND_MAX - 0.5) * spacing;
				tempZ = ((double)rand() / RAND_MAX - 0.5) * spacing;

				Trajectory::Ptr newTraj = Trajectory::create();
				newTraj->gap = 0;

				for (int k = 0; k < trajs[j]->particles.size(); ++k) {
					Particle::Ptr newPart = Particle::create();
					newPart->id = trajs[j]->particles[k]->id + i * trajs[j]->particles.size();
					newPart->frame_index = trajs[j]->particles[k]->frame_index;
					newPart->x = trajs[j]->particles[k]->x + tempX;
					newPart->y = trajs[j]->particles[k]->y + tempY;
					newPart->z = trajs[j]->particles[k]->z + tempZ;

					newTraj->particles.push_back(newPart);
				}

				tempTrajs.push_back(newTraj);
			}
		}
		return tempTrajs;
	}
	else
		return trajs;

}

void DataSetGenerator::read3DTrajectoryFile(string filename, lpt::InputFormat format, int timesOfTrajs) {

	switch (format) {
	case lpt::BINARY: //TODO: ADD BINARY FILE SUPPORT FOR TRAJ INPUT
		cout << "BINARY format support needs to be coded for trajectory input" <<endl; //TODO: add YAML support
		break;
	case lpt::PLAINTEXT:
		{
			lpt::Input in;
			vector<lpt::Trajectory::Ptr> originalTrajs = in.trajinput( filename.c_str() );
			vector<lpt::Trajectory::Ptr> trajs = generateAdditionalTrajectories(timesOfTrajs, originalTrajs);
			cout << "(datagen.cpp) Number of Trajectories = " << trajs.size() << endl;
			for (int i = 0; i < trajs.size(); ++i){
				//polyFitTraj(*trajs[i], 3, 0.005);
				for (int j = 0; j < trajs[i]->particles.size(); ++j){
					double x = trajs[i]->particles[j]->x;
					double y = trajs[i]->particles[j]->y;
					double z = trajs[i]->particles[j]->z;
					int id = trajs[i]->particles[j]->id;
					int frame_index = trajs[i]->particles[j]->frame_index;
					cv::Point3d newparticle(x,y,z);
					this->frames[frame_index].first.push_back(id);             //vector of particle ids
					this->frames[frame_index].second.push_back(newparticle);   //vector of particles (cv::Point3f)

					// if there is only one particle in traj, insert dummy particles in mid and end points vec
					if (trajs[i]->particles.size() <= 1) {
						this->frames_midpoints[frame_index].first.push_back(id);
						this->frames_midpoints[frame_index].second.push_back(newparticle);
						
						this->frames_endpoints[frame_index].first.push_back(id);
						this->frames_endpoints[frame_index].second.push_back(newparticle);

						break; // exit this loop
					}

					// calculate velocities for x y and z
					double u, v, w;
					double mid_x, mid_y, mid_z;
					double end_x, end_y, end_z;

					double maxPixelVelocity = 10; // the maximum movement per interval is 10 pixels.

					double timeInterval = 0.005; // hard code here, 0.005s.
					if (j == 0) {
						u = (trajs[i]->particles[j + 1]->x - trajs[i]->particles[j]->x) / timeInterval;
						v = (trajs[i]->particles[j + 1]->y - trajs[i]->particles[j]->y) / timeInterval;
						w = (trajs[i]->particles[j + 1]->z - trajs[i]->particles[j]->z) / timeInterval;

						mid_x = x + u * 0.5 * timeInterval;
						mid_y = y + v * 0.5 * timeInterval;
						mid_z = z + w * 0.5 * timeInterval;

						this->frames_midpoints[frame_index].first.push_back(id);
						this->frames_midpoints[frame_index].second.push_back(cv::Point3d(mid_x, mid_y, mid_z));

						end_x = trajs[i]->particles[j + 1]->x;
						end_y = trajs[i]->particles[j + 1]->y;
						end_z = trajs[i]->particles[j + 1]->z;

						this->frames_endpoints[frame_index].first.push_back(id);
						this->frames_endpoints[frame_index].second.push_back(cv::Point3d(end_x, end_y, end_z));
					}
					else if (j == trajs[i]->particles.size() - 1) {
						u = (trajs[i]->particles[j]->x - trajs[i]->particles[j-1]->x) / timeInterval;
						v = (trajs[i]->particles[j]->y - trajs[i]->particles[j-1]->y) / timeInterval;
						w = (trajs[i]->particles[j]->z - trajs[i]->particles[j-1]->z) / timeInterval;

						mid_x = x + u * 0.5 * timeInterval;
						mid_y = y + v * 0.5 * timeInterval;
						mid_z = z + w * 0.5 * timeInterval;

						this->frames_midpoints[frame_index].first.push_back(id);
						this->frames_midpoints[frame_index].second.push_back(cv::Point3d(mid_x, mid_y, mid_z));

						end_x = mid_x + u * 0.5 * timeInterval;
						end_y = mid_y + v * 0.5 * timeInterval;
						end_z = mid_z + w * 0.5 * timeInterval;

						this->frames_endpoints[frame_index].first.push_back(id);
						this->frames_endpoints[frame_index].second.push_back(cv::Point3d(end_x, end_y, end_z));
					}
					else {
						u = (trajs[i]->particles[j+1]->x - trajs[i]->particles[j - 1]->x) /(2 * timeInterval);
						v = (trajs[i]->particles[j+1]->y - trajs[i]->particles[j - 1]->y) /(2 * timeInterval);
						w = (trajs[i]->particles[j+1]->z - trajs[i]->particles[j - 1]->z) /(2 * timeInterval);

						mid_x = x + u * 0.5 * timeInterval;
						mid_y = y + v * 0.5 * timeInterval;
						mid_z = z + w * 0.5 * timeInterval;

						this->frames_midpoints[frame_index].first.push_back(id);
						this->frames_midpoints[frame_index].second.push_back(cv::Point3d(mid_x, mid_y, mid_z));

						end_x = trajs[i]->particles[j + 1]->x;
						end_y = trajs[i]->particles[j + 1]->y;
						end_z = trajs[i]->particles[j + 1]->z;

						this->frames_endpoints[frame_index].first.push_back(id);
						this->frames_endpoints[frame_index].second.push_back(cv::Point3d(end_x, end_y, end_z));
					}
				}
			}
		}
		break;
	case lpt::YAMLTEXT: //TODO: ADD YAML FILE SUPPORT FOR TRAJ INPUT
		cout << "YAML format support needs to be coded for trajectory input" <<endl; //TODO: add YAML support
		break;
	default:
		cout << "3D trajectory format selected is not supported" << endl;
		break;
	}
}

void DataSetGenerator::project3DFramesTo2D() {
	auto& cameras = this->shared_objects->cameras;
	for (int camera_index = 0; camera_index < cameras.size(); ++camera_index ) {
		lpt::Camera& cam = cameras[camera_index];
		cv::Mat camera_matrix;
		cv::Mat rotation_matrix;
		cv::Mat translation_vec;
		cv::Mat dist_coeffs;
		lpt::convertCamParameters2CV(cam, camera_matrix, dist_coeffs,
				rotation_matrix, translation_vec);

		cv::Mat rotation_vec;
		cv::Rodrigues(rotation_matrix, rotation_vec);
		array<double, 3> p_camera, p_world;
		
		//p_camera[0] = cam.sensor_size[0] / 2;
		//p_camera[0] = cam.sensor_size[1] / 2;
		//p_camera[0] = 0.0;
		p_camera[0] = cam.sensor_size[0] / 2;
		p_camera[1] = cam.sensor_size[1] / 2;
		p_camera[2] = 0.0;

		lpt::convertCameraCoordinatesToWorld(cam, p_camera, p_world);
		double focal_length = cam.f[0] * cam.pixel_size[0];

		//cv::namedWindow("temp", cv::WINDOW_AUTOSIZE);

		map<int, pair<vector<int>, vector<cv::Point3f> > >::iterator frame_it;
		for( frame_it = frames.begin(); frame_it != frames.end(); ++frame_it ) {
			int frame_index = frame_it->first;
			
			vector<int>& point_ids = frame_it->second.first;     // this is a vector<int> particleIDs
			vector<cv::Point3f>& points3D = frame_it->second.second;  // this is a vector<cv::Point3f> frame of 3D particles
			vector<cv::Point2f> image_points;
			
			cv::projectPoints(cv::Mat(points3D), rotation_vec, translation_vec,
				camera_matrix, dist_coeffs, image_points);

			// vector of 3D & 2D midpoints
			vector<cv::Point3f>& midpoints3D = frames_midpoints[frame_it->first].second;
			vector<cv::Point2f> image_midpoints;

			cv::projectPoints(cv::Mat(midpoints3D), rotation_vec, translation_vec,
				camera_matrix, dist_coeffs, image_midpoints);

			// vector of 3D & 2D endpoints
			vector<cv::Point3f>& endpoints3D = frames_endpoints[frame_it->first].second;
			vector<cv::Point2f> image_endpoints;

			cv::projectPoints(cv::Mat(endpoints3D), rotation_vec, translation_vec,
				camera_matrix, dist_coeffs, image_endpoints);

			lpt::ImageFrame newframe(frame_index);
		
			//lpt::convertFrame(image_points, newframe, point_ids);

			for (int p = 0; p < image_points.size(); ++p) {
				double distance = 
					sqrt(
						  (points3D[p].x - p_world[0]) * (points3D[p].x - p_world[0]) 
						+ (points3D[p].y - p_world[1]) * (points3D[p].y - p_world[1]) 
						+ (points3D[p].z - p_world[2]) * (points3D[p].z - p_world[2]) 
					);
	
				//cout << distance << endl;

				double radius = focal_length * image_creator->object_size / ( abs( distance - focal_length ) ) / 2.0 / cam.pixel_size[0] * 0.5;  // the "*0.5" was because of too large particles
				//double radius = 2.0;
				if (radius < 1.0 && radius >= 0.5) 
					radius = 0.5;
				double intensity = 150.0;//image_creator->object_intensity / ( distance * distance );
				double pi = 3.141592653;
				double random_n = rand() / (RAND_MAX + 1.0);
				double sigma = 1;
				intensity = intensity + intensity * 1 / sigma / sqrt(2 * pi) * exp(-random_n * random_n / (2 * sigma * sigma));
				//cout << "radius = " << radius << " intensity = " << intensity << " distance " << distance << endl;
				lpt::ParticleImage::Ptr newparticle = lpt::ParticleImage::create(point_ids[p], image_points[p].x, image_points[p].y, radius, intensity);
				newframe.particles.push_back(newparticle);

				//lpt::ParticleStreakImage::Ptr newstreak = lpt::ParticleStreakImage::create(point_ids[p], radius, intensity);
				//lpt::ParticleStreakImage::Ptr newstreak_float = lpt::ParticleStreakImage::create(point_ids[p], radius, intensity);

				//newstreak->curvePoints.push_back(cv::Point(image_points[p]));
				//newstreak->curvePoints.push_back(cv::Point(image_midpoints[p]));
				//newstreak->curvePoints.push_back(cv::Point(image_endpoints[p]));
				//newframe.streaks.push_back(newstreak);

				//newstreak_float->curvePoints2f.push_back(image_points[p]);
				//newstreak_float->curvePoints2f.push_back(image_midpoints[p]);
				//newstreak_float->curvePoints2f.push_back(image_endpoints[p]);
				//newframe.streaks_floats.push_back(newstreak_float);
			}
			
			//!!!!!!!!!!!!!!!
			//std::random_shuffle(newframe.particles.begin(), newframe.particles.end()); 
			//!!!!!!!!!!!!!!!
			
			// this line is used to create streak images instead of circle images
			//image_creator->createStreaksImage(newframe);
			//image_creator->createBlurStreamsImage(newframe);


			
			//image_creator->createImage(newframe); // rember to comment this line when generating streak image
			
			image_creator->createGaussianImage(newframe); // rember to comment this line when generating streak image

			//cv::imshow("temp", newframe.image);
			//cv::waitKey(0);
			// 

			cam.frames.push_back(newframe);
		}
    }
	frames.clear(); 
	frames.swap(map<int, pair<vector<int>, vector<cv::Point3f> > >());
}

void DataSetGenerator::writeImageFramePoints(string data_path, string basename) {
	ofstream fout;
	auto& cameras = this->shared_objects->cameras;
	for (int c = 0; c < cameras.size(); ++c){
		YAML::Emitter out;
		out << cameras[c].frames;
		stringstream framesfile;
		framesfile << data_path << cameras[c].id << basename;
		fout.open( framesfile.str().c_str() );
		fout << out.c_str();
		fout.close();
	}
}

void DataSetGenerator::writeCameraPairs(string filename) {
	ofstream fout;
	auto& camera_pairs = this->shared_objects->camera_pairs;

	YAML::Emitter pairs_out;
	pairs_out << camera_pairs;
	fout.open( filename.c_str() );
	fout << pairs_out.c_str();
	fout.close();
}

void DataSetGenerator::createSpiralTrajectories(
        int number_of_frames, int number_of_particles,
        int d, double theta)
{  /* TODO: port matlab code here
    vector<Frame> frames(number_of_frames);
    for (int f; f < frames.size(); ++f){
        double xo, yo, zo;

        Particle newParticle;
        newParticle.x =
        newParticle.y =
        newParticle.z =
        frames.particles

    }
   */
}

void DataSetGenerator::showImages() {
	auto& cameras = this->shared_objects->cameras;
	stringstream capturedetails;
	for (int i = 0; i < cameras.size(); i++) {
		for (int f = 0; f < cameras[i].frames.size(); f++) {
			capturedetails.str("");
			capturedetails << cameras[i].name << "  --  Frame " << f;

			for (int p = 0; p < cameras[i].frames[f].particles.size(); p++) {
				auto particle = cameras[i].frames[f].particles[p];
				//cv::circle(cameras[i].frames[f].image, cv::Point(static_cast<int>(particle->x), static_cast<int>(particle->y)),
				//	static_cast<int>(particle->radius * 2), 200, 1);
			}

			cv::imshow("show", cameras[i].frames[f].image);
			cv::displayStatusBar("show", capturedetails.str(), 1);
			cv::waitKey(1);
		}
	}
}

void DataSetGenerator::createOpenFOAMTrajectories(int number_of_frames, int number_of_particles){
	//TODO: Incorporate OpenFOAM simulation to create trajectories
}


DataSetGenerator::~DataSetGenerator() {
    // TODO Auto-generated destructor stub
}

} /* NAMESPACE_PT */
