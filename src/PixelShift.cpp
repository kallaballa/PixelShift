#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <sndfile.hh>
#include <cmath>
#include <chrono>
#include <stdlib.h>
#include <time.h>
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
using std::chrono::microseconds;

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

void render(const std::vector<double>& absSpectrum, Mat& img, const size_t& i) {
		Mat frame = img.clone();
		Mat hsvImg;
		cvtColor(img, hsvImg, CV_RGB2HSV);

		uint8_t rot = rand() % 255;

	for (int h = 0; h < img.rows; h++) {
			for (int w = 0; w < img.cols; w++) {
				auto& vec = img.at<Vec3b>(h,w);
				auto& vech = hsvImg.at<Vec3b>(h,w);
				uint16_t hue = (((uint16_t)vech[0]) + rot) % 255;
				assert(hue <= 255);
				vech[0] = (((hue + rot) % 255) / 64) * 64;
				vech[2] = (vech[2] / 64) * 64;

				double hsvradian = ((double)vech[2] / 255.0) * 2.0 * M_PI;
				double vx = cos(hsvradian);
		    double vy = sin(hsvradian);

		    double x = w + (((vx * vech[2]) * absSpectrum[vech[2] % 4]) / 30);
		    double y = h + (((vy * vech[2]) * absSpectrum[vech[2] % 4]) / 30);

		    if(x >= 0 && y >= 0 && x < frame.cols && y < frame.rows) {
		    	auto& vecf = frame.at<Vec3b>(y,x);
		    	vecf[0] = vec[0];
		    	vecf[1] = vec[1];
		    	vecf[2] = vec[2];
				}
			}
		}

		Mat blur;
		GaussianBlur(frame,blur, {0,0}, 1, 1);

		double factor = 0.5;
		for (int h = 0; h < img.rows; h++) {
			for (int w = 0; w < img.cols; w++) {
	    	auto& vec = img.at<Vec3b>(h,w);
	    	auto& vecb = blur.at<Vec3b>(h,w);
				auto& vecf = frame.at<Vec3b>(h,w);

				vecf[0] = lerp(factor, vecb[0], vec[0]);
				vecf[1] = lerp(factor, vecb[1], vec[1]);
				vecf[2] = lerp(factor, vecb[2], vec[2]);
			}
		}

		std::string zeroes = "000000000";
		std::string num = std::to_string(i + 1);
		num = zeroes.substr(0, zeroes.length() - num.length()) + num;
		imwrite("frame" + num + ".jpg", frame);
}

void pixelShift(VideoCapture& capture, SndfileHandle& file, size_t fps) {
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

			render(absSpectrum, frame, i);
		}
	}
}

int main(int argc, char** argv) {
	srand (time(NULL));
	SndfileHandle sndfile(argv[1]);
  VideoCapture capture(argv[2]);
  if( !capture.isOpened() )
      throw "Error when reading " + std::string(argv[2]);
	pixelShift(capture, sndfile, 25);

	return 0;
}
