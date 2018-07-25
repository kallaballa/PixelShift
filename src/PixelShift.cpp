#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include <boost/program_options.hpp>
#include <sndfile.hh>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#define cimg_use_jpeg
#include "aquila/global.h"
#include "aquila/functions.h"
#include "aquila/transform/FftFactory.h"
#include "aquila/source/FramesCollection.h"
#include "aquila/tools/TextPlot.h"

using namespace cv;
using std::vector;
using std::chrono::microseconds;

namespace po = boost::program_options;

typedef unsigned char sample_t;


std::pair<std::vector<KeyPoint>, std::vector<KeyPoint>> generate_keypoints(Mat& testRGB, Mat& goalRGB) {
  Mat img_1;
  Mat img_2;

  cvtColor(goalRGB, img_1, CV_RGB2GRAY);
  cvtColor(testRGB, img_2, CV_RGB2GRAY);

  cv::Ptr<cv::ORB> detector = cv::ORB::create(5000);
  cv::Ptr<cv::ORB> extractor = cv::ORB::create();

   std::vector<KeyPoint> keypoints_1, keypoints_2;

   detector->detect( img_1, keypoints_1 );
   detector->detect( img_2, keypoints_2 );
   return {keypoints_1, keypoints_2};
}

cv::Mat PointVec2HomogeneousMat(const std::vector<cv::Point2f>& pts)
{
	int num_pts = pts.size();
	cv::Mat homMat(3, num_pts, CV_32FC1);
	for(int i=0; i<num_pts; i++){
		homMat.at<float>(0,i) = pts[i].x;
		homMat.at<float>(1,i) = pts[i].y;
		homMat.at<float>(2,i) = 1.0;
	}
	return homMat;
}


// Morph points
void MorphPoints(const std::vector<cv::Point2f>& srcPts1, const std::vector<cv::Point2f>& srcPts2, std::vector<cv::Point2f>& dstPts, float s = 0.5)
{
	assert(srcPts1.size() == srcPts2.size());

	int num_pts = srcPts1.size();

	dstPts.resize(num_pts);
	for(int i=0; i<num_pts; i++){
		dstPts[i].x = (1.0 - s) * srcPts1[i].x + s * srcPts2[i].x;
		dstPts[i].y = (1.0 - s) * srcPts1[i].y + s * srcPts2[i].y;
	}
}


void GetTriangleVertices(const cv::Subdiv2D& sub_div, const std::vector<cv::Point2f>& points, std::vector<cv::Vec3i>& triangle_vertices)
{
	std::vector<cv::Vec6f> triangles;
	sub_div.getTriangleList(triangles);

	int num_triangles = triangles.size();
	triangle_vertices.clear();
	triangle_vertices.reserve(num_triangles);
	for(int i=0; i<num_triangles; i++){
		std::vector<cv::Point2f>::const_iterator vert1, vert2, vert3;
		vert1 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][0],triangles[i][1]));
		vert2 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][2],triangles[i][3]));
		vert3 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][4],triangles[i][5]));

		cv::Vec3i vertex;
		if(vert1 != points.end() && vert2 != points.end() && vert3 != points.end()){
			vertex[0] = vert1 - points.begin();
			vertex[1] = vert2 - points.begin();
			vertex[2] = vert3 - points.begin();
			triangle_vertices.push_back(vertex);
		}
	}
}


void TransTrianglerPoints(const std::vector<cv::Vec3i>& triangle_vertices,
	const std::vector<cv::Point2f>& points,
	std::vector<std::vector<cv::Point2f>>& triangler_pts)
{
	int num_triangle = triangle_vertices.size();
	triangler_pts.resize(num_triangle);
	for(int i=0; i<num_triangle; i++){
		std::vector<cv::Point2f> triangle;
		for(int j=0; j<3; j++){
			triangle.push_back(points[triangle_vertices[i][j]]);
		}
		triangler_pts[i] = triangle;
	}
}


void PaintTriangles(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& triangles)
{
	int num_triangle = triangles.size();

	for(int i=0; i<num_triangle; i++){
		std::vector<cv::Point> poly(3);

		for(int j=0;j<3;j++){
			poly[j] = cv::Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		cv::fillConvexPoly(img, poly,  cv::Scalar(i+1));
	}
}

///// for debug /////
void DrawTriangles(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& triangles)
{
	int num_triangle = triangles.size();

	std::vector<std::vector<cv::Point>> polies;
	for(int i=0; i<num_triangle; i++){
		std::vector<cv::Point> poly(3);

		for(int j=0;j<3;j++){
			poly[j] = cv::Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		polies.push_back(poly);
	}
	cv::polylines(img, polies, true, cv::Scalar(255,0,255));
}
//////////////////////

void SolveHomography(const std::vector<cv::Point2f>& src_pts1, const std::vector<cv::Point2f>& src_pts2, cv::Mat& H)
{
	assert(src_pts1.size() == src_pts2.size());

	H = PointVec2HomogeneousMat(src_pts2) * PointVec2HomogeneousMat(src_pts1).inv();
}


void SolveHomography(const std::vector<std::vector<cv::Point2f>>& src_pts1,
	const std::vector<std::vector<cv::Point2f>>& src_pts2,
	std::vector<cv::Mat>& Hmats)
{
	assert(src_pts1.size() == src_pts2.size());

	int pts_num = src_pts1.size();
	Hmats.clear();
	Hmats.reserve(pts_num);
	for(int i=0; i<pts_num; i++){
		cv::Mat H;
		SolveHomography(src_pts1[i], src_pts2[i], H);
		Hmats.push_back(H);
	}
}


// Morph homography matrix
void MorphHomography(const cv::Mat& Hom, cv::Mat& MorphHom1, cv::Mat& MorphHom2, float blend_ratio)
{
	cv::Mat invHom = Hom.inv();
	MorphHom1 = cv::Mat::eye(3,3,CV_32FC1) * (1.0 - blend_ratio) + Hom * blend_ratio;
	MorphHom2 = cv::Mat::eye(3,3,CV_32FC1) * blend_ratio + invHom * (1.0 - blend_ratio);
}



// Morph homography matrix
void MorphHomography(const std::vector<cv::Mat>& Homs,
	std::vector<cv::Mat>& MorphHoms1,
	std::vector<cv::Mat>& MorphHoms2,
	float blend_ratio)
{
	int hom_num = Homs.size();
	MorphHoms1.resize(hom_num);
	MorphHoms2.resize(hom_num);
	for(int i=0; i<hom_num; i++){
		MorphHomography(Homs[i], MorphHoms1[i], MorphHoms2[i], blend_ratio);
	}
}


// create a map for cv::remap()
void CreateMap(const cv::Mat& TriangleMap, const std::vector<cv::Mat>& HomMatrices, cv::Mat& map_x, cv::Mat& map_y)
{
	assert(TriangleMap.type() == CV_32SC1);

	// Allocate cv::Mat for the map
	map_x.create(TriangleMap.size(), CV_32FC1);
	map_y.create(TriangleMap.size(), CV_32FC1);

	// Compute inverse matrices
	std::vector<cv::Mat> invHomMatrices(HomMatrices.size());
	for(size_t i=0; i<HomMatrices.size(); i++){
		invHomMatrices[i] = HomMatrices[i].inv();
	}

	for(int y=0; y<TriangleMap.rows; y++){
		for(int x=0; x<TriangleMap.cols; x++){
			int idx = TriangleMap.at<int>(y,x)-1;
			if(idx >= 0){
				cv::Mat H = invHomMatrices[TriangleMap.at<int>(y,x)-1];
				float z = H.at<float>(2,0) * x + H.at<float>(2,1) * y + H.at<float>(2,2);
				if(z==0)
					z = 0.00001;
				map_x.at<float>(y,x) = (H.at<float>(0,0) * x + H.at<float>(0,1) * y + H.at<float>(0,2)) / z;
				map_y.at<float>(y,x) = (H.at<float>(1,0) * x + H.at<float>(1,1) * y + H.at<float>(1,2)) / z;
			}
			else{
				map_x.at<float>(y,x) = x;
				map_y.at<float>(y,x) = y;
			}
		}
	}
}


//! Image Morphing
/*!
\param[in] src_img1 Input image 1
\param[in] src_points1 Points on the image 1
\param[in] src_img2 Input image 2
\param[in] src_points2 Points on the image 2, which must be corresponded to src_point1
\param[out] dst_img Morphed output image
\param[out] dst_points Morphed points on the output image
\param[in] shape_ratio blending ratio (0.0 - 1.0) of shape between image 1 and 2.  If it is 0.0, output shape is same as src_img1.
\param[in] color_ratio blending ratio (0.0 - 1.0) of color between image 1 and 2.  If it is 0.0, output color is same as src_img1. If it is negative, it is set to shape_ratio.
*/
void ImageMorphing(const cv::Mat& src_img1, const std::vector<cv::Point2f>& src_points1,
	const cv::Mat& src_img2, const std::vector<cv::Point2f>& src_points2,
	cv::Mat& dst_img, std::vector<cv::Point2f>& dst_points,
	float shape_ratio = 0.5, float color_ratio = -1)
{
	// Input Images
	cv::Mat SrcImg[2];
	SrcImg[0] = src_img1;
	SrcImg[1] = src_img2;

	// Input Points
	std::vector<cv::Point2f> SrcPoints[2];
	SrcPoints[0].insert(SrcPoints[0].end(), src_points1.begin(), src_points1.end());
	SrcPoints[1].insert(SrcPoints[1].end(), src_points2.begin(), src_points2.end());

	// Add 4 corner points of image to the points
	cv::Size img_size[2];
	for(int i=0; i<2; i++){
		img_size[i] = SrcImg[i].size();
		float w = img_size[i].width - 1;
		float h= img_size[i].height - 1;
		SrcPoints[i].push_back(cv::Point2f(0,0));
		SrcPoints[i].push_back(cv::Point2f(w,0));
		SrcPoints[i].push_back(cv::Point2f(0,h));
		SrcPoints[i].push_back(cv::Point2f(w,h));
	}

	// Morph points
	std::vector<cv::Point2f> MorphedPoints;
	MorphPoints(SrcPoints[0], SrcPoints[1], MorphedPoints, shape_ratio);

	// Generate Delaunay Triangles from the morphed points
	int num_points = MorphedPoints.size();
	cv::Size MorphedImgSize(MorphedPoints[num_points-1].x+1,MorphedPoints[num_points-1].y+1);
	cv::Subdiv2D sub_div(cv::Rect(0,0,MorphedImgSize.width,MorphedImgSize.height));
	sub_div.insert(MorphedPoints);

	// Get the ID list of corners of Delaunay traiangles.
	std::vector<cv::Vec3i> triangle_indices;
	GetTriangleVertices(sub_div, MorphedPoints, triangle_indices);

	// Get coordinates of Delaunay corners from ID list
	std::vector<std::vector<cv::Point2f>> triangle_src[2], triangle_morph;
	TransTrianglerPoints(triangle_indices, SrcPoints[0], triangle_src[0]);
	TransTrianglerPoints(triangle_indices, SrcPoints[1], triangle_src[1]);
	TransTrianglerPoints(triangle_indices, MorphedPoints, triangle_morph);

	// Create a map of triangle ID in the morphed image.
	cv::Mat triangle_map = cv::Mat::zeros(MorphedImgSize, CV_32SC1);
	PaintTriangles(triangle_map, triangle_morph);

	// Compute Homography matrix of each triangle.
	std::vector<cv::Mat> homographyMats, MorphHom[2];
	SolveHomography(triangle_src[0], triangle_src[1], homographyMats);
	MorphHomography(homographyMats, MorphHom[0], MorphHom[1], shape_ratio);

	cv::Mat trans_img[2];
	for(int i=0; i<2; i++){
		// create a map for cv::remap()
		cv::Mat trans_map_x, trans_map_y;
		CreateMap(triangle_map, MorphHom[i], trans_map_x, trans_map_y);

		// remap
		cv::remap(SrcImg[i], trans_img[i], trans_map_x, trans_map_y, cv::INTER_LINEAR);
	}

	// Blend 2 input images
	float blend = (color_ratio < 0) ? shape_ratio : color_ratio;
	dst_img = trans_img[0] * (1.0 - blend) + trans_img[1] * blend;

	dst_points.clear();
	dst_points.insert(dst_points.end(), MorphedPoints.begin(), MorphedPoints.end() - 4);

}

double mix_channels(vector<double>& buffer) {
	double avg = 0;
	for (double& d : buffer) {
		avg += d;
	}
	avg /= buffer.size();

	return avg;
}

std::vector<double> read_fully(SndfileHandle& file, size_t channels) {
	std::vector<double> data;
	vector<double> buffer(channels);

	while (file.read(buffer.data(), channels) > 0) {
		data.push_back(mix_channels(buffer));
	}

	return data;
}

uint8_t lerp(double factor, uint8_t a, uint8_t b) {
	return factor * a + (1.0 - factor) * b;
}


void cannyThreshold(Mat& src, Mat& 	detected_edges) {
	Mat src_gray;
	cvtColor( src, src_gray, CV_RGB2GRAY );
  /// Reduce noise with a kernel 3x3
  GaussianBlur( src_gray, detected_edges, Size(3,3), 1, 1 );

  /// Canny detector
  Canny( detected_edges, detected_edges, 0, 255, 3 );
 }


void render(VideoWriter& output, const std::vector<double>& absSpectrum,
		Mat& sourceRGB, const size_t& iteration, const double& boost,
		size_t tweens, const size_t& component, const bool& randomizeDir, const bool& edgeDetect, const bool& zeroout, const bool& morph) {
	std::vector<Mat> tweenVec(tweens);
	Mat hsvImg;
	Mat sourceRGBA;
	cvtColor(sourceRGB, hsvImg, CV_RGB2HSV);
	cvtColor(sourceRGB, sourceRGBA, CV_RGB2RGBA);
	Mat edges;
	size_t morphTweens = 0;
	if(morph) {
		morphTweens = tweens;
		tweens = 1;
	}
	if(edgeDetect)
		cannyThreshold(sourceRGB,edges);

	uint8_t rot = 0;
	if (randomizeDir)
		rot = rand() % 255;

	for (size_t t = 0; t < tweens; ++t) {
		Mat& tween = tweenVec[t];
		tween = Mat(sourceRGBA.rows, sourceRGBA.cols, sourceRGBA.type());

		if(!edgeDetect) {
			if(zeroout)
				tween = Scalar::all(0);
		}
		else {
			if(zeroout)
				tween = Scalar::all(0);
			else
				tween = sourceRGBA.clone();
		}
		for (int h = 0; h < sourceRGBA.rows; h++) {
			for (int w = 0; w < sourceRGBA.cols; w++) {
				if(edgeDetect && !edges.at<uint8_t>(h,w))
					continue;

				auto& vec = sourceRGBA.at<Vec4b>(h, w);
				auto& vech = hsvImg.at<Vec3b>(h, w);
				uint16_t hue = (((uint16_t) vech[0]) + rot) % 255;
				assert(hue <= 255);
				vech[0] = (((uint8_t) hue) / 32) * 32;
				vech[2] = (vech[2] / 32) * 32;

				uint8_t mod = vech[component];
				double hsvradian = ((double) mod / 255.0) * 2.0 * M_PI;
				double vx = cos(hsvradian);
				double vy = sin(hsvradian);

				double x = w
						+ ((((vx * mod) * absSpectrum[mod % 8]) / (100 / boost)) / (t + 1));
				double y = h
						+ ((((vy * mod) * absSpectrum[mod % 8]) / (100 / boost)) / (t + 1));

				if (x >= 0 && y >= 0 && x < tween.cols && y < tween.rows) {
					auto& vect = tween.at<Vec4b>(y, x);
					vect[0] = vec[0];
					vect[1] = vec[1];
					vect[2] = vec[2];
					vect[3] = vec[3];
				}
			}
		}
	}
	tweens = morphTweens;
	Mat frame = sourceRGBA.clone();
	std::vector<Mat> blurVec(tweens);

	if(morph) {
		for (size_t i = 0; i < tweens; ++i) {
			Mat trgb;
			cvtColor(tweenVec[0], trgb, CV_RGBA2RGB);
			auto kpp = generate_keypoints(sourceRGB, trgb);
			std::vector<Point2f> kp1;
			std::vector<Point2f> kp2;
			std::vector<Point2f> outputkp;

			for(KeyPoint& kp : kpp.first) {
				kp1.push_back(kp.pt);
			}
			for(KeyPoint& kp : kpp.second) {
				kp2.push_back(kp.pt);
			}

			ImageMorphing(sourceRGB,kp1,trgb, kp2, blurVec[i], outputkp, 1.0/tweens, 1.0/tweens);
		}
	} else {
		for (size_t i = 0; i < tweens; ++i) {
			if(edgeDetect) {
				blurVec[i] = tweenVec[i].clone();
			} else {
					GaussianBlur(tweenVec[i], blurVec[i], { 0, 0 }, 1, 1);
			}
		}
	}


	double factor = 1.0 / tweens;
	for (size_t i = 0; i < tweens; ++i) {
		for (int h = 0; h < sourceRGBA.rows; h++) {
			for (int w = 0; w < sourceRGBA.cols; w++) {
				auto& vecb = blurVec[i].at<Vec4b>(h, w);
				auto& vecf = frame.at<Vec4b>(h, w);

				if(vecb[3] > 0) {
					vecf[0] = lerp(factor, vecb[0], vecf[0]);
					vecf[1] = lerp(factor, vecb[1], vecf[1]);
					vecf[2] = lerp(factor, vecb[2], vecf[2]);
				}
			}
		}
	}
//	imshow("", frame);
//	waitKey(10);
	Mat frameRGB;
	cvtColor(frame, frameRGB, CV_RGBA2RGB);
	output.write(frameRGB);
}

void pixelShift(VideoCapture& capture, SndfileHandle& file, VideoWriter& output,
		size_t fps, double boost, size_t tweens, size_t component,
		bool randomizeDir, bool edgeDetect, bool zeroout, bool morph) {
	Mat frame;
	double samplingRate = file.samplerate();
	size_t channels = file.channels();
	vector<double> data = read_fully(file, channels);
	size_t i = 0;
	using namespace Aquila;
	const std::size_t SIZE = round((double) samplingRate / (double) fps);

	SignalSource in(data, samplingRate);
	FramesCollection frames(in, SIZE);
	SpectrumType filterSpectrum(SIZE);
	auto signalFFT = FftFactory::getFft(16);

	auto start = std::chrono::system_clock::now();
	size_t f = 0;
#pragma omp for ordered schedule(dynamic)
	for (size_t j = 0; j < frames.count(); ++j) {
		bool success;
#pragma omp ordered
		{
			success = capture.read(frame);
			++i;
		}

		if (success) {
			SpectrumType spectrum = signalFFT->fft(frames.frame(j).toArray());
			std::size_t halfLength = spectrum.size() / 2;
			std::vector<double> absSpectrum(halfLength);
			for (std::size_t k = 0; k < halfLength; ++k) {
				absSpectrum[k] = std::abs(spectrum[k]);
			}

			render(output, absSpectrum, frame, i, boost, tweens, component,
					randomizeDir, edgeDetect, zeroout, morph);
			if(f == 10) {
				auto duration = std::chrono::duration_cast<microseconds>(
							std::chrono::system_clock::now() - start);
				std::cerr << ((double)f / ((double)duration.count() / 1000000))  << std::endl;
				start = std::chrono::system_clock::now();
				f = 0;
			} else {
				++f;
			}
		}
	}
}

int main(int argc, char** argv) {
	using std::string;
	srand(time(NULL));
	string videoFile;
	string audioFile;
	string outputVideo = "output.mkv";
	size_t fps = 25;
	double boost = 1;
	size_t tweens = 3;
	size_t component = 0;
	bool randomizeDir = false;
	bool edgeDetect = false;
	bool zeroout = false;
	bool morph = false;
	po::options_description genericDesc("Options");
	genericDesc.add_options()
			("fps,f",	po::value<size_t>(&fps)->default_value(fps), "The frame rate of the resulting video")
			("boost,b",	po::value<double>(&boost)->default_value(boost),"Boost factor for the effect. Higher values boost more and values below 1 dampen")
			("tweens,t", po::value<size_t>(&tweens)->default_value(tweens), "How many in between steps should the effect produce")
			("output,o", po::value<string>(&outputVideo)->default_value(outputVideo),	"The filename of the resulting video")
			("morph,m",	"Use image morphing for tweening")
			("hue,h",	"Use the hue of the picture to steer the effect")
			("sat,s",	"Use the saturation of the picture to steer the effect")
			("val,v", "Use the value of the picture to steer the effect")
			("edge,e","Use edge detection to limit the effect")
			("rand,r","Randomize the direction of the effect")
			("zero,z","Zero out tweens before transformation")
			("help", "Produce help message");

	po::options_description hidden("Hidden options");
	hidden.add_options()("audioFile", po::value<string>(&audioFile), "audioFile");
	hidden.add_options()("videoFile", po::value<string>(&videoFile), "videoFile");

	po::options_description cmdline_options;
	cmdline_options.add(genericDesc).add(hidden);

	po::positional_options_description p;
	p.add("audioFile", 1);
	p.add("videoFile", 1);

	po::options_description visible;
	visible.add(genericDesc);

	po::variables_map vm;
	po::store(
			po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(),
			vm);
	po::notify(vm);

	if (vm.count("help") || audioFile.empty() || videoFile.empty()) {
		std::cerr << "Usage: pixelshift [options] <audioFile> <videoFile>"
				<< std::endl;
		std::cerr << visible;
		return 0;
	}

	if ((vm.count("hue") && vm.count("sat"))
			|| (vm.count("hue") && vm.count("val"))
			|| (vm.count("sat") && vm.count("val"))) {
		std::cerr << "Only one of hue, sat or val may be specified" << std::endl;
	}

	if (vm.count("hue")) {
		component = 0;
	} else if (vm.count("sat")) {
		component = 1;
	} else if (vm.count("val")) {
		component = 2;
	}

	if (vm.count("rand")) {
		randomizeDir = true;
	}

	if (vm.count("edge")) {
		edgeDetect = true;
	}

	if (vm.count("zero")) {
		zeroout = true;
	}

	if (vm.count("morph")) {
		morph = true;
	}

	SndfileHandle sndfile(audioFile);
	VideoCapture capture(videoFile);
	double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter output(outputVideo, CV_FOURCC('H', '2', '6', '4'), fps,
			Size(width, height));

	if (!capture.isOpened())
		throw "Error when reading " + videoFile;

	pixelShift(capture, sndfile, output, fps, boost, tweens, component,
			randomizeDir, edgeDetect, zeroout, morph);
	capture.release();
	output.release();
	return 0;
}
