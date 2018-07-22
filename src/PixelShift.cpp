#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

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

void render(const std::vector<double>& absSpectrum, Mat& source,
		const size_t& iteration, const size_t& dampen, const size_t& tweens, const bool& randomizeDir) {
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

				uint8_t mod = vech[2];
				double hsvradian = ((double) mod / 255.0) * 2.0 * M_PI;
				double vx = cos(hsvradian);
				double vy = sin(hsvradian);

				double x = w + ((((vx * mod) * absSpectrum[mod % 8]) / dampen) / (t + 1));
				double y = h + ((((vy * mod) * absSpectrum[mod % 8]) / dampen) / (t + 1));

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
	std::string zeroes = "000000000";
	std::string num = std::to_string(iteration + 1);
	num = zeroes.substr(0, zeroes.length() - num.length()) + num;
	imwrite("frame" + num + ".jpg", frame);
}

void pixelShift(VideoCapture& capture, SndfileHandle& file, size_t fps, size_t dampen, size_t tweens, bool randomizeDir) {
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

			render(absSpectrum, frame, i, dampen, tweens, randomizeDir);
		}
	}
}

int main(int argc, char** argv) {
	srand (time(NULL));
	SndfileHandle sndfile(argv[1]);
  VideoCapture capture(argv[2]);
  if( !capture.isOpened() )
      throw "Error when reading " + std::string(argv[2]);
	pixelShift(capture, sndfile, 25, 20, 3, true);

	return 0;
}
