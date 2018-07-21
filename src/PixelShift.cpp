#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <sndfile.hh>
#include <cmath>
#define cimg_use_jpeg
#include "CImg.h"
#include "aquila/global.h"
#include "aquila/functions.h"
#include "aquila/transform/FftFactory.h"
#include "aquila/source/FramesCollection.h"
#include "aquila/tools/TextPlot.h"

using namespace cimg_library;
using std::vector;
typedef unsigned char sample_t;
typedef CImg<sample_t> image_t;


typedef struct RgbColor {
	sample_t r;
	sample_t g;
	sample_t b;
} RgbColor;

typedef struct HsvColor {
	sample_t h;
	sample_t s;
	sample_t v;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv) {
	RgbColor rgb;
	sample_t region, remainder, p, q, t;

	if (hsv.s == 0) {
		rgb.r = hsv.v;
		rgb.g = hsv.v;
		rgb.b = hsv.v;
		return rgb;
	}

	region = hsv.h / 43;
	remainder = (hsv.h - (region * 43)) * 6;

	p = (hsv.v * (255 - hsv.s)) >> 8;
	q = (hsv.v * (255 - ((hsv.s * remainder) >> 8))) >> 8;
	t = (hsv.v * (255 - ((hsv.s * (255 - remainder)) >> 8))) >> 8;

	switch (region) {
	case 0:
		rgb.r = hsv.v;
		rgb.g = t;
		rgb.b = p;
		break;
	case 1:
		rgb.r = q;
		rgb.g = hsv.v;
		rgb.b = p;
		break;
	case 2:
		rgb.r = p;
		rgb.g = hsv.v;
		rgb.b = t;
		break;
	case 3:
		rgb.r = p;
		rgb.g = q;
		rgb.b = hsv.v;
		break;
	case 4:
		rgb.r = t;
		rgb.g = p;
		rgb.b = hsv.v;
		break;
	default:
		rgb.r = hsv.v;
		rgb.g = p;
		rgb.b = q;
		break;
	}

	return rgb;
}

HsvColor RgbToHsv(RgbColor rgb) {
	HsvColor hsv;
	sample_t rgbMin, rgbMax;

	rgbMin =
			rgb.r < rgb.g ?
					(rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
	rgbMax =
			rgb.r > rgb.g ?
					(rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

	hsv.v = rgbMax;
	if (hsv.v == 0) {
		hsv.h = 0;
		hsv.s = 0;
		return hsv;
	}

	hsv.s = 255 * long(rgbMax - rgbMin) / hsv.v;
	if (hsv.s == 0) {
		hsv.h = 0;
		return hsv;
	}

	if (rgbMax == rgb.r)
		hsv.h = 0 + 43 * (rgb.g - rgb.b) / (rgbMax - rgbMin);
	else if (rgbMax == rgb.g)
		hsv.h = 85 + 43 * (rgb.b - rgb.r) / (rgbMax - rgbMin);
	else
		hsv.h = 171 + 43 * (rgb.r - rgb.g) / (rgbMax - rgbMin);

	return hsv;
}

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

void render(const std::vector<double>& absSpectrum, const image_t& img, const size_t& i) {
		image_t frame(img.width(), img.height(), img.depth(), img.spectrum());
		frame.fill((sample_t)0,0,0,0);
		for (int h = 0; h < img.height(); h++) {
			for (int w = 0; w < img.width(); w++) {
				RgbColor rgb = {img(w, h, 0, 0), img(w, h, 0, 1), img(w, h, 0, 2)};
				HsvColor hsv = RgbToHsv(rgb);
				uint16_t hue = (((uint16_t)hsv.h) + (255 / ((i % 25) + 1))) % 255;
				assert(hue <= 255);
				hsv.h = (hue / 64) * 64;
				uint8_t rb = (rgb.b / 64) * 64;


				double vx = cos(((double)hsv.h / 255.0) * 2.0 * M_PI);
		    double vy = sin(((double)hsv.h / 255.0) * 2.0 * M_PI);

		    double x = w + (((vx * rb) * absSpectrum[hsv.h % 8]) / 10);
				double y = h + (((vy * rb) * absSpectrum[hsv.h % 8]) / 10);

				if(x >= 0 && y >= 0 && x < frame.width() && y < frame.height()) {
					frame(x, y, 0, 0) = rgb.r;
					frame(x, y, 0, 1) = rgb.g;
					frame(x, y, 0, 2) = rgb.b;
				}
			}
		}
		std::string zeroes = "000000000";
		std::string num = std::to_string(i + 1);
		num = zeroes.substr(0, zeroes.length() - num.length()) + num;
		frame.save(("frame" + num + ".jpg").c_str());
}

void pixelShift(SndfileHandle& file, const image_t& img, size_t fps) {
  double samplingRate = file.samplerate();
  size_t channels = file.channels();
  vector<double> data = read_fully(file, channels);
  size_t i = 0;
  using namespace Aquila;
  const std::size_t SIZE = round((double)samplingRate / (double)fps);

  SignalSource in(data, samplingRate);
  FrequencyType sampleFreq = samplingRate;

  FramesCollection frames(in,SIZE);
  SpectrumType filterSpectrum(SIZE);
  auto signalFFT = FftFactory::getFft(16);

  #pragma omp for schedule(dynamic)
  for (size_t j = 0; j < frames.count(); ++j) {
    SpectrumType spectrum = signalFFT->fft(frames.frame(j).toArray());

    std::size_t halfLength = spectrum.size() / 2;
    std::vector<double> absSpectrum(halfLength);
    for (std::size_t i = 0; i < halfLength; ++i)
    {
        absSpectrum[i] = std::abs(spectrum[i]);
    }
    render(absSpectrum, img, i);
    ++i;
  }
}

int main(int argc, char** argv) {
	SndfileHandle sndfile(argv[1]);
	image_t img(argv[2]);
	pixelShift(sndfile, img, 25);

	return 0;
}
