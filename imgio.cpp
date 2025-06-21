#include "imgio.h"

#include "grain_newson_et_al/film_grain_rendering.h"
#include <opencv2/xphoto.hpp>


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
    if (cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1) {
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

cv::Mat exponentialBlur(const cv::Mat& img, float sigma, int kernel_radius = -1) {
    CV_Assert(sigma > 0);
    // Determine radius
    if (kernel_radius < 0) {
        kernel_radius = static_cast<int>(3 * sigma);
    }
    kernel_radius = std::max(1, kernel_radius);
    int ksize = 2 * kernel_radius + 1;

    // Build exponential kernel (CV_32F)
    cv::Mat kernel(ksize, ksize, CV_32F);
    int center = kernel_radius;
    for (int y = 0; y < ksize; y++) {
        int dy = y - center;
        for (int x = 0; x < ksize; x++) {
            int dx = x - center;
            float r = std::sqrt(float(dx*dx + dy*dy));
            kernel.at<float>(y, x) = std::exp(-r / sigma);
        }
    }
    // Normalize
    kernel /= cv::sum(kernel)[0];

    // Prepare 8-bit image for filtering
    cv::Mat img8u;
    if (img.depth() == CV_32F) {
        // scale [0,1]→[0,255]
        img.convertTo(img8u, CV_8U, 255.0);
    } else if (img.depth() == CV_8U) {
        img8u = img;
    } else {
        // convert other depths to float first, then scale
        cv::Mat tmp;
        img.convertTo(tmp, CV_32F);
        tmp.convertTo(img8u, CV_8U, 255.0 / (std::numeric_limits<float>::max()));
        // Note: if input isn’t [0,1], result may be unexpected.
    }

    // Apply filter; filter2D handles multi-channel automatically
    cv::Mat blurred8u;
    cv::filter2D(img8u, blurred8u, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

    // Convert back to float [0,1]
    cv::Mat out;
    blurred8u.convertTo(out, CV_32F, 1.0f / 255.0f);
    return out;
}

cv::Mat applyHalation(const cv::Mat& in, float intensity, float radius) {
    // Apply these steps:
    //  1. Threshold
    //  2. Create a mask of where you did not threshold
    //  3. Blur the thresholded image
    //  4. Mask the blurred image by where you did not threshold
    //  5. Apply the blurred image back to the original

    cv::Mat thresholded;
    cv::threshold(in, thresholded, 0.7f, 1.0f, cv::THRESH_TOZERO);

    cv::Mat gray;
    cv::cvtColor(thresholded, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_8U, 255.0f);

    cv::Mat binary;
    cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);

    cv::Mat invBinary;
    cv::bitwise_not(binary, invBinary);

    cv::Mat blurred = exponentialBlur(thresholded, radius);

    // Applying halation only on the edges works sometimes but other times can leave some nasty-looking artifacts near
    //  the edges. I should add this back if I notice that lights are unnaturally bright. If I do add this back, I
    //  should add a "soft mask" instead of a binary thing
    // cv::Mat halationOnly;
    // cv::bitwise_and(blurred, blurred, halationOnly, invBinary);

    // Multiply to give halation the redshift
    cv::Mat tinted;
    cv::multiply(blurred, cv::Scalar(0.02f, 0.05f, 1.0f), tinted);

    cv::Mat result;
    cv::addWeighted(in, 1.0f, tinted, intensity, 0.0f, result);

    return result;
}

cv::Mat denoise(const cv::Mat& in) {
    /*
     * This denoising process converts the image from linear to rec709 - then back to linear when complete. At first
     * this doesn't make sense since digital noise is in linear space. The reason I do this anyway is because, under the
     * hood, dctDenoising converts the image to the uint8 format. If the image is kept in linear color space, deep
     * shadows get compressed to the point of having an extreme color banding effect when it is converted back to rec709.
     * The colors may also be off in some images. Denoising in rec709 fixes this issue.
     */
    cv::Mat rec = linearToRec709(in);
    cv::Mat rec255;
    rec.convertTo(rec255, CV_8U, 255.0f);  // having some segfault problems so converting explicitly can't hurt

    cv::Mat denoised255;
    cv::xphoto::dctDenoising(rec255, denoised255, 7.5, 8);

    cv::Mat out;
    denoised255.convertTo(out, CV_32FC3, 1.0f / 255.0f);
    return rec709toLinear(out);
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

cv::Mat addGrainColor(const cv::Mat& in) {
    float muR = 0.08;
    float sigmaR = 0.005;
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

cv::Mat addGrainMonochrome(const cv::Mat& in) {
    cv::Mat monochrome;
    cv::cvtColor(in, monochrome, cv::COLOR_BGR2GRAY);

    float muR = 0.11;
    float sigmaR = 0.005;
    float s = 1.0;
    float sigmaFilter = 0.8;
    unsigned int NmonteCarlo = 350;
    float xA = 0;
    float yA = 0;
    auto xB = static_cast<float>(monochrome.cols);
    auto yB = static_cast<float>(monochrome.rows);
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

    // Process each channel separately
    auto *imgIn = new matrix<float>();
    imgIn->allocate_memory((int) monochrome.rows, (int) monochrome.cols);

    for (unsigned int i = 0; i < (unsigned int) monochrome.rows; i++) {
        for (unsigned int j = 0; j < (unsigned int) monochrome.cols; j++) {
            const auto& pixel = monochrome.at<float>((int) i, (int) j);
            imgIn->set_value((int) i, (int) j, pixel);
        }
    }

    options.grainSeed = static_cast<unsigned int>(std::time(nullptr));

    matrix<float>* imgOutTemp = film_grain_rendering_pixel_wise(imgIn, options);

    cv::Mat grain(imgOutTemp->get_nrows(), imgOutTemp->get_ncols(), CV_32FC1, imgOutTemp->get_ptr());

    cv::Mat difference;
    cv::subtract(grain, monochrome, difference, cv::noArray(), CV_32FC1);
    cv::Mat difference3;
    cv::cvtColor(difference, difference3, cv::COLOR_GRAY2BGR);

    // Reduce grain in shadows
    const float n = 0.5;
    cv::Mat grainWeight;
    cv::pow(monochrome, 1.0f / 3.0f, grainWeight);
    grainWeight = grainWeight * n - (1 - n);

    cv::Mat grainWeight3;
    cv::cvtColor(grainWeight, grainWeight3, cv::COLOR_GRAY2BGR);

    cv::Mat weightedGrain;
    cv::multiply(difference3, grainWeight3, weightedGrain);  // result is CV_32FC3

    cv::Mat output;
    cv::addWeighted(in, 1.0f, weightedGrain, 1.0f, 0.0f, output);  // output is CV_32FC3
    delete imgOutTemp;  // TODO: should I delete imgIn?

    output = cv::max(cv::min(output, 1.0f), 0.0f);  // Clamp [0, 1]

    return output;
}

void save(const std::string& filepath, const cv::Mat& img) {
    cv::Mat saveImg;
    img.convertTo(saveImg, CV_8U, 255.0f);
    bool ok = cv::imwritemulti(filepath, saveImg);
    if (!ok) {
        std::cerr << "WARNING: Could not write to " << filepath << "\n";
    }
}
