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
    show("Frame", img, 800);

    std::string lutPath = "luts/Apple LOG to Fujifilm 3513DI D60 Rec709 G2-4.cube";
    applyLUT(img, img, lutPath);
    show("LUT applied", img, 800);

    cv::Mat linear = rec709toLinear(img);
    show("Rec709 -> Linear", linear, 800);
    show("Rec709 -> Linear -> Rec709", linearToRec709(linear), 800);

    cv::Mat grainyColor = addGrain(img);
    show("Grainy Color", grainyColor, 800);

    return 0;
}
