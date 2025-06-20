#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <OpencolorIO/OpenColorIO.h>

#include "imgio.h"

namespace ocio = OCIO_NAMESPACE;

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);

    cv::Mat img = loadFirstFrame("hammock_landscape.mov");
//    show("Frame", img, 800);
    save("out/original.png", img);

    std::string lutPath = "luts/Apple LOG to Fujifilm 3513DI D60 Rec709 G2-4.cube";
    cv::Mat lut;
    applyLUT(img, lut, lutPath);
//    show("LUT applied", lut, 800);
    save("out/lut.png", lut);

    cv::Mat linear = rec709toLinear(lut);
//    show("Linear", linear, 800);
    save("out/linear.png", linear);

    cv::Mat halation = applyHalation(linear, 0.2f, 15.0f);
    save("out/halation.png", halation);

    cv::Mat denoised;
    denoise(halation, denoised, true);
//    show("Denoised", denoised, 800);
    save("out/denoised.png", denoised);

    cv::Mat grainyColor = addGrain(denoised);
//    show("Grainy Color", grainyColor, 800);
    save("out/grainy.png", grainyColor);

    cv::Mat result = linearToRec709(grainyColor);
    save("out/result.png", result);

    return 0;
}
