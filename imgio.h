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
cv::Mat applyHalation(const cv::Mat& in, float intensity, float radius);
cv::Mat denoise(const cv::Mat& in);
cv::Mat rec709toLinear(const cv::Mat& in);
cv::Mat linearToRec709(const cv::Mat& img);
cv::Mat addGrainColor(const cv::Mat& in);
cv::Mat addGrainMonochrome(const cv::Mat& in);
void save(const std::string& filepath, const cv::Mat& img);

#endif  // FILM_EMULATION_CPP_IMGIO_H
