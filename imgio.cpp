#include "imgio.h"

#include "grain_newson_et_al/film_grain_rendering.h"

cv::Mat loadFirstFrame(const std::string& path) {
    cv::VideoCapture cap{path};
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video");
    }

    cv::Mat frame;
    if (!cap.read(frame)) {
        throw std::runtime_error("Could not read frame");
    }

    frame.convertTo(frame, CV_32F, 1.0f/255.0f);

    return frame;
}

void applyProcessor(const cv::Mat& in, cv::Mat& out, const ocio::ConstProcessorRcPtr& processor) {
    // Assumptions about the input image:
    //  1. It is in BGR format
    //  2. It is normalized to [0, 1] and has the dtype of floats
    //  3. It has only three channels - no alpha (an extension of point 1)
    cv::Mat tmp;
    cv::cvtColor(in, tmp, cv::COLOR_BGR2RGB);
    if (!tmp.isContinuous()) {
        tmp = tmp.clone();
    }

    ocio::PackedImageDesc imgDesc(
            static_cast<void*>(tmp.ptr<float>()),
            tmp.cols, tmp.rows, tmp.channels()
    );
    ocio::ConstCPUProcessorRcPtr cpu = processor->getDefaultCPUProcessor();
    cpu->apply(imgDesc);

    cv::min(tmp, 1.0f, tmp);  // Prevent >1
    cv::cvtColor(tmp, tmp, cv::COLOR_RGB2BGR);

    out = tmp;
}

void show(const std::string& name, const cv::Mat& img, uint32_t height) {
    // Resize while preserving aspect ratio
    double scale = static_cast<double>(height) / img.rows;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), scale, scale);
    cv::imshow(name, resized);
    cv::waitKey(0);
    if (cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 0) {
        cv::destroyWindow(name);
    }
}

void applyLUT(const cv::Mat& in, cv::Mat& out, const std::string& lutPath) {
    auto config = ocio::Config::CreateRaw();
    ocio::FileTransformRcPtr ft = ocio::FileTransform::Create();
    ft->setSrc(lutPath.c_str());
    ft->setInterpolation(ocio::INTERP_LINEAR);

    auto processor = config->getProcessor(ft);
    applyProcessor(in, out, processor);
}

cv::Mat rec709toLinear(const cv::Mat& img) {
    cv::Mat linear(img.size(), CV_32FC3);

    cv::Mat mask = img < 0.081f;

    cv::divide(img, 4.5f, linear, 1.0, CV_32FC3);
    linear.setTo(0, ~mask); // zero out non-mask areas

    cv::Mat imgNonmask;
    img.copyTo(imgNonmask, ~mask);

    cv::Mat temp;
    cv::add(imgNonmask, 0.099f, temp);
    cv::divide(temp, 1.099f, temp);
    cv::pow(temp, 1.0f / 0.45f, temp);

    temp.copyTo(linear, ~mask);

    return linear;
}

cv::Mat linearToRec709(const cv::Mat& img) {
    cv::Mat rec709(img.size(), CV_32FC3);

    cv::Mat mask = img < 0.018f;
    cv::multiply(img, 4.5f, rec709, 1, CV_32FC3);
    rec709.setTo(0, ~mask);

    cv::Mat imgNonmask;
    img.copyTo(imgNonmask, ~mask);

    cv::Mat temp;
    cv::pow(imgNonmask, 0.45f, temp);
    cv::multiply(temp, 1.099f, temp);
    cv::subtract(temp, 0.099f, temp);

    temp.copyTo(rec709, ~mask);
    return rec709;
}

cv::Mat addGrain(const cv::Mat& in) {
    float muR = 0.05;
    float sigmaR = 0.01;
    float s = 1.0;
    float sigmaFilter = 0.8;
    unsigned int NmonteCarlo = 175;
    float xA = 0;
    float yA = 0;
    auto xB = static_cast<float>(in.cols);
    auto yB = static_cast<float>(in.rows);
    unsigned int mOut = (int) floor(s * (yB-yA));
    unsigned int nOut = (int) floor(s * (xB-xA));

    filmGrainOptionsStruct<float> options{
            .muR = muR,
            .sigmaR = sigmaR,
            .sigmaFilter = sigmaFilter,
            .NmonteCarlo = NmonteCarlo,
            .algorithmID = 0,
            .s = s,
            .xA = xA,
            .yA = yA,
            .xB = xB,
            .yB = yB,
            .mOut = mOut,
            .nOut = nOut,
            .grainSeed = 0
    };

    cv::Mat grain(in.rows, in.cols, CV_32FC3);

    // Process each channel separately
    auto *imgIn = new matrix<float>();
    for (int colorChannel = 0; colorChannel < 3; colorChannel++) {
        imgIn->allocate_memory((int) in.rows, (int) in.cols);

        for (unsigned int i = 0; i < (unsigned int) in.rows; i++) {
            for (unsigned int j = 0; j < (unsigned int) in.cols; j++) {
                const auto& pixel = in.at<cv::Vec3f>((int) i, (int) j);
                imgIn->set_value((int) i, (int) j, pixel[colorChannel]);
            }
        }

        options.grainSeed = static_cast<unsigned int>(std::time(nullptr)) + colorChannel;

        matrix<float>* imgOutTemp = film_grain_rendering_pixel_wise(imgIn, options);

        cv::Mat grainyChannel(imgOutTemp->get_nrows(), imgOutTemp->get_ncols(),
                              CV_32FC1, imgOutTemp->get_ptr());

        for (int i = 0; i < grain.rows; i++) {
            for (int j = 0; j < grain.cols; j++) {
                grain.at<cv::Vec3f>(i, j)[colorChannel] = grainyChannel.at<float>(i, j);
            }
        }

        // TODO: should I delete imgIn?
        delete imgOutTemp;
    }

    return grain;
}
