#ifndef FILM_EMULATION_CPP_IMGIO_H
#define FILM_EMULATION_CPP_IMGIO_H

#include <opencv2/opencv.hpp>
#include <OpenColorIO/OpenColorIO.h>
#include <string>

namespace ocio = OCIO_NAMESPACE;

cv::Mat loadFirstFrame(const std::string& path);
void applyProcessor(const cv::Mat& in, cv::Mat& out, const ocio::ConstProcessorRcPtr& processor);
void show(const std::string& name, const cv::Mat& img, uint32_t height);
void applyLUT(const cv::Mat& in, cv::Mat& out, const std::string& lutPath);
cv::Mat rec709toLinear(const cv::Mat& in);
cv::Mat linearToRec709(const cv::Mat& img);
cv::Mat addGrain(const cv::Mat& in);

#endif  // FILM_EMULATION_CPP_IMGIO_H
