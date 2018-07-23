#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <boost/program_options.hpp>
#include <sndfile.hh>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define cimg_use_jpeg
#include "aquila/global.h"
#include "aquila/functions.h"
#include "aquila/transform/FftFactory.h"
#include "aquila/source/FramesCollection.h"
#include "aquila/tools/TextPlot.h"

using namespace cv;
using std::vector;
namespace po = boost::program_options;

typedef unsigned char sample_t;

double mix_channels(vector<double>& buffer) {
  double avg = 0;
  for(double& d: buffer) {
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
	return factor*a + (1.0 - factor)*b;
}

void render(VideoWriter& output, const std::vector<double>& absSpectrum, Mat& source,
		const size_t& iteration, const double& boost, const size_t& tweens, const size_t& component, const bool& randomizeDir) {
	std::vector<Mat> tweenVec(tweens);
	Mat hsvImg;
	cvtColor(source, hsvImg, CV_RGB2HSV);

	uint8_t rot = 0;
	if(randomizeDir)
		rot = rand() % 255;

	for (size_t t = 0; t < tweens; ++t) {
		Mat& tween = tweenVec[t];
		tween = source.clone();
		for (int h = 0; h < source.rows; h++) {
			for (int w = 0; w < source.cols; w++) {
				auto& vec = source.at<Vec3b>(h, w);
				auto& vech = hsvImg.at<Vec3b>(h, w);
				uint16_t hue = (((uint16_t) vech[0]) + rot) % 255;
				assert(hue <= 255);
				vech[0] = (((uint8_t) hue) / 32) * 32;
				vech[2] = (vech[2] / 32) * 32;

				uint8_t mod = vech[component];
				double hsvradian = ((double) mod / 255.0) * 2.0 * M_PI;
				double vx = cos(hsvradian);
				double vy = sin(hsvradian);

				double x = w + ((((vx * mod) * absSpectrum[mod % 8]) / (100 / boost)) / (t + 1));
				double y = h + ((((vy * mod) * absSpectrum[mod % 8]) / (100 / boost)) / (t + 1));

				if (x >= 0 && y >= 0 && x < tween.cols && y < tween.rows) {
					auto& vect = tween.at<Vec3b>(y, x);
					vect[0] = vec[0];
					vect[1] = vec[1];
					vect[2] = vec[2];
				}
			}
		}
	}

	std::vector<Mat> blurVec(tweens);
	for(size_t i = 0; i < tweens; ++i) {
		GaussianBlur(tweenVec[i], blurVec[i], { 0, 0 }, 1, 1);
	}

	Mat frame = source.clone();
	double factor = 1.0/tweens;
	for(size_t i = 0; i < tweens; ++i) {
		for (int h = 0; h < source.rows; h++) {
			for (int w = 0; w < source.cols; w++) {
				auto& vecb = blurVec[i].at<Vec3b>(h, w);
				auto& vecf = frame.at<Vec3b>(h, w);

				vecf[0] = lerp(factor, vecb[0], vecf[0]);
				vecf[1] = lerp(factor, vecb[1], vecf[1]);
				vecf[2] = lerp(factor, vecb[2], vecf[2]);
			}
		}
	}
	output.write(frame);
}

void pixelShift(VideoCapture& capture, SndfileHandle& file, VideoWriter& output, size_t fps, double boost, size_t tweens, size_t component, bool randomizeDir) {
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

			render(output, absSpectrum, frame, i, boost, tweens, component, randomizeDir);
		}
	}
}

int main(int argc, char** argv) {
	using std::string;
	srand (time(NULL));
	string videoFile;
	string audioFile;
	string outputVideo = "output.mkv";
	size_t fps = 25;
	double boost = 1;
	size_t tweens = 3;
	size_t component = 0;
	bool randomizeDir = false;

  po::options_description genericDesc("Options");
  genericDesc.add_options()
  		("fps,f", po::value<size_t>(&fps)->default_value(fps),"The frame rate of the resulting video")
      ("boost,b", po::value<double>(&boost)->default_value(boost), "Boost factor for the effect. Higher values boost more and values below 1 dampen")
      ("tweens,t", po::value<size_t>(&tweens)->default_value(tweens), "How many in between steps should the effect produce")
      ("output,o", po::value<string>(&outputVideo)->default_value(outputVideo), "The filename of the resulting video")
			("hue,h", "Use the hue of the picture to steer the effect")
      ("sat,s", "Use the saturation of the picture to steer the effect")
      ("val,v", "Use the value of the picture to steer the effect")
      ("rand,r", "Randomize the direction of the effect")
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
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help") || audioFile.empty() || videoFile.empty()) {
    std::cerr << "Usage: pixelshift [options] <audioFile> <videoFile>" << std::endl;
    std::cerr << visible;
    return 0;
  }

  if((vm.count("hue") && vm.count("sat")) || (vm.count("hue") && vm.count("val")) || (vm.count("sat") && vm.count("val"))) {
  	std::cerr << "Only one of hue, sat or val may be specified" << std::endl;
  }

  if(vm.count("hue")) {
  	component = 0;
  } else if(vm.count("sat")) {
  	component = 1;
  } else if(vm.count("val")) {
  	component = 2;
  }

  if(vm.count("rand")) {
  	randomizeDir = true;
  }


	SndfileHandle sndfile(audioFile);
  VideoCapture capture(videoFile);
  double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  VideoWriter output(outputVideo,CV_FOURCC('F','F','V','1'),fps, Size(width,height));

  if( !capture.isOpened() )
      throw "Error when reading " + std::string(argv[2]);
	pixelShift(capture, sndfile, output, fps, boost, tweens, component, randomizeDir);
	capture.release();
	output.release();
	return 0;
}
