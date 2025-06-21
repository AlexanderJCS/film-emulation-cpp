#include <iostream>
#include <chrono>  // for timing
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <OpencolorIO/OpenColorIO.h>

#include "imgio.h"

namespace ocio = OCIO_NAMESPACE;
namespace fs = std::filesystem;


bool hasMovExtension(const fs::path& path) {
    if (!path.has_extension()) return false;
    std::string ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), ::tolower);  // convert to lowercase
    return ext == ".mov";
}


std::vector<fs::path> getMovFiles(const fs::path& directory) {
    std::vector<fs::path> movFiles;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && hasMovExtension(entry.path())) {
            movFiles.push_back(entry.path());
        }
    }

    return movFiles;
}


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);

    for (const fs::path& movFile : getMovFiles(fs::path("./in"))) {
        std::cout << "Processing: " << movFile << std::endl;
        std::string filename = movFile.stem().string();

        auto tStart = std::chrono::high_resolution_clock::now();
        cv::Mat img = loadFirstFrame(movFile);
        //    show("Frame", img, 800);
        save("out/" + filename + "_1_original.png", img);

        std::string lutPath = "luts/Apple LOG to Fujifilm 3513DI D60 Rec709 G2-4.cube";
        cv::Mat lut;
        applyLUT(img, lut, lutPath);
        //    show("LUT applied", lut, 800);
        save("out/" + filename + "_2_lut.png", lut);

        cv::Mat linear = rec709toLinear(lut);
        //    show("Linear", linear, 800);
        save("out/" + filename + "_3_linear.png", linear);

        cv::Mat halation = applyHalation(linear, 0.3f, 20.0f);
        save("out/" + filename + "_4_halation.png", halation);

        cv::Mat denoised;
        denoise(halation, denoised, true);
        //    show("Denoised", denoised, 800);
        save("out/" + filename + "_5_denoised.png", denoised);

        cv::Mat grain = addGrainMonochrome(denoised);
        //    show("Grain", grain, 800);
        save("out/" + filename + "_6_grainy.png", grain);

        cv::Mat result = linearToRec709(grain);
        save("out/_" + filename + "_7_result.png", result);

        auto tEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = tEnd - tStart;
        std::cout << "Completed in " << elapsed.count() << " seconds\n";
    }

    return 0;
}
