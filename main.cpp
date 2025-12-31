#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>
#include <limits>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// Factory for detectors
cv::Ptr<cv::Feature2D> makeDetector(const std::string& name, int nfeatures = 2000) {
    if (name == "SIFT") return cv::SIFT::create();
    else if (name == "ORB") return cv::ORB::create(nfeatures);
    else if (name == "AKAZE") return cv::AKAZE::create();
    else throw std::runtime_error("Unknown detector: " + name);
}

// Choose matcher depending on descriptor type
cv::Ptr<cv::DescriptorMatcher> makeMatcher(const std::string& detector_name) {
    if (detector_name == "SIFT") return cv::FlannBasedMatcher::create();
    else if (detector_name == "ORB" || detector_name == "AKAZE")
        return cv::BFMatcher::create(cv::NORM_HAMMING);
    else throw std::runtime_error("Unknown detector: " + detector_name);
}

// Convert path to imread-friendly string
std::string path_to_string(const fs::path& p) { return p.string(); }

int main(int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    fs::path image_folder = fs::path("../../images");
    std::string image_left_name, image_right_name, detector_name;
    double scale;
    bool use_ratio_test;
    float ratio_thresh;
    double ransac_reproj_thresh;
    std::string input;

    // Filenames
    std::cout << "Enter filename of left image (without extension): ";
    std::cin >> image_left_name;
    if (image_left_name.find(".jpg") == std::string::npos)
        image_left_name += ".jpg";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << "Enter filename of right image (without extension): ";
    std::cin >> image_right_name;
    if (image_right_name.find(".jpg") == std::string::npos)
        image_right_name += ".jpg";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Detector
    std::cout << "Choose detector (SIFT | ORB | AKAZE) [default: SIFT]: ";
    std::getline(std::cin, input);
    detector_name = input.empty() ? "SIFT" : input;

    // Scale
    std::cout << "Enter scale factor (e.g., 1.0 for full size, 0.5 for half size) [default: 0.25]: ";
    std::getline(std::cin, input);
    scale = input.empty() ? 0.25 : std::stod(input);

    // Ratio test
    std::cout << "Use ratio test? (1 = yes, 0 = no) [default: 1]: ";
    std::getline(std::cin, input);
    use_ratio_test = input.empty() ? true : (std::stoi(input) != 0);

    // Lowe's ratio
    std::cout << "Enter Lowe's ratio threshold [default: 0.75]: ";
    std::getline(std::cin, input);
    ratio_thresh = input.empty() ? 0.75f : std::stof(input);

    // RANSAC threshold
    std::cout << "Enter RANSAC reprojection threshold in pixels [default: 3.0]: ";
    std::getline(std::cin, input);
    ransac_reproj_thresh = input.empty() ? 3.0 : std::stod(input);

    const fs::path left_path = image_folder / image_left_name;
    const fs::path right_path = image_folder / image_right_name;

    std::cout << "\n--- Configuration ---\n";
    std::cout << "Left image:  " << left_path << "\n";
    std::cout << "Right image: " << right_path << "\n";
    std::cout << "Detector: " << detector_name
        << " | Ratio test: " << (use_ratio_test ? "ON" : "OFF") << "\n";
    std::cout << "Scale: " << scale
        << " | Ratio threshold: " << ratio_thresh
        << " | RANSAC threshold: " << ransac_reproj_thresh << "\n";
    std::cout << "----------------------\n\n";

    // Load images
    cv::Mat img1 = cv::imread(path_to_string(left_path), cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(path_to_string(right_path), cv::IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load one or both images." << std::endl;
        std::cerr << "Left image path:  " << fs::absolute(left_path) << " (exists: " << fs::exists(left_path) << ")" << std::endl;
        std::cerr << "Right image path: " << fs::absolute(right_path) << " (exists: " << fs::exists(right_path) << ")" << std::endl;
        return 1;
    }

    if (scale != 1.0) {
        cv::resize(img1, img1, cv::Size(), scale, scale);
        cv::resize(img2, img2, cv::Size(), scale, scale);
    }

    // Create stored_images folder
    const fs::path stored_folder = image_folder / "stored_images";
    if (!fs::exists(stored_folder)) fs::create_directory(stored_folder);

    // Detector & matcher
    cv::Ptr<cv::Feature2D> detector;
    try { detector = makeDetector(detector_name); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    cv::Ptr<cv::DescriptorMatcher> matcher = makeMatcher(detector_name);

    // Detect and compute
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    detector->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

    std::cout << "Detected keypoints: img1=" << kpts1.size()
        << " img2=" << kpts2.size() << "\n";

    // Match descriptors
    std::cout << "Matching descriptors..." << std::endl;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);
    std::cout << "Found " << knn_matches.size() << " knn matches." << std::endl;

    std::vector<cv::DMatch> good_matches;
    if (use_ratio_test) {
        for (const auto& mvec : knn_matches)
            if (mvec.size() >= 2 && mvec[0].distance < ratio_thresh * mvec[1].distance)
                good_matches.push_back(mvec[0]);
    }
    else {
        for (const auto& mvec : knn_matches)
            if (!mvec.empty()) good_matches.push_back(mvec[0]);
    }

    if (good_matches.size() < 4) {
        std::cerr << "Not enough matches.\n"; return 1;
    }
    std::cout << "Good matches after ratio test: " << good_matches.size() << "\n";

    // Compute homography
    std::cout << "Computing homography..." << std::endl;
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }

    std::vector<unsigned char> inliers_mask;
    cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, ransac_reproj_thresh, inliers_mask);
    
    // Count inliers
    int num_inliers = 0;
    for (auto m : inliers_mask) {
        if (m) num_inliers++;
    }
    std::cout << "RANSAC inliers: " << num_inliers << " / " << good_matches.size() << std::endl;
    std::cout << "Homography computed. Warping images..." << std::endl;

    // Create feature matching visualization with statistics
    std::vector<char> inliers_char(inliers_mask.begin(), inliers_mask.end());
    cv::Mat match_img;
    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, match_img, 
                    cv::Scalar::all(-1), cv::Scalar::all(-1), 
                    inliers_char, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Create statistics overlay
    int text_height = 30;
    int stats_height = 250;
    cv::Mat stats_img(stats_height + match_img.rows, match_img.cols, CV_8UC3, cv::Scalar(40, 40, 40));
    match_img.copyTo(stats_img(cv::Rect(0, stats_height, match_img.cols, match_img.rows)));
    
    // Add text statistics
    int y_offset = 35;
    cv::putText(stats_img, "PANORAMA STITCHING STATISTICS", cv::Point(20, y_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    y_offset += 40;
    
    cv::putText(stats_img, "Detector: " + detector_name, cv::Point(20, y_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(100, 255, 100), 1);
    y_offset += 30;
    
    cv::putText(stats_img, "Left Image: " + image_left_name + "  |  Right Image: " + image_right_name, 
                cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    y_offset += 30;
    
    cv::putText(stats_img, "Keypoints - Left: " + std::to_string(kpts1.size()) + 
                "  |  Right: " + std::to_string(kpts2.size()), 
                cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 100), 1);
    y_offset += 30;
    
    cv::putText(stats_img, "Matches: " + std::to_string(knn_matches.size()) + 
                "  |  Good Matches (ratio=" + std::to_string(ratio_thresh) + "): " + std::to_string(good_matches.size()), 
                cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 100), 1);
    y_offset += 30;
    
    cv::putText(stats_img, "RANSAC Inliers: " + std::to_string(num_inliers) + 
                " (" + std::to_string((int)(100.0 * num_inliers / good_matches.size())) + "%)" +
                "  |  Threshold: " + std::to_string(ransac_reproj_thresh) + "px", 
                cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(100, 255, 255), 1);
    y_offset += 30;
    
    cv::putText(stats_img, "Scale: " + std::to_string(scale), 
                cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // Warp images
    std::vector<cv::Point2f> corners2 = { {0,0}, {(float)img2.cols,0}, {(float)img2.cols,(float)img2.rows}, {0,(float)img2.rows} };
    std::vector<cv::Point2f> corners2_warped;
    cv::perspectiveTransform(corners2, corners2_warped, H);

    std::vector<cv::Point2f> all_corners = { {0,0}, {(float)img1.cols,0}, {(float)img1.cols,(float)img1.rows}, {0,(float)img1.rows} };
    all_corners.insert(all_corners.end(), corners2_warped.begin(), corners2_warped.end());

    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
    for (auto& p : all_corners) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    cv::Mat translation = (cv::Mat_<double>(3, 3) << 1, 0, -min_x, 0, 1, -min_y, 0, 0, 1);
    cv::Mat H_translated = translation * H;

    int pano_w = (int)std::ceil(max_x - min_x);
    int pano_h = (int)std::ceil(max_y - min_y);

    cv::Mat warped1(pano_h, pano_w, CV_8UC3, cv::Scalar::all(0));
    cv::Mat warped2;
    img1.copyTo(warped1(cv::Rect((int)-min_x, (int)-min_y, img1.cols, img1.rows)));
    cv::warpPerspective(img2, warped2, H_translated, cv::Size(pano_w, pano_h));

    // Masks (single-channel)
    cv::Mat mask1(warped1.size(), CV_8U, cv::Scalar(0));
    cv::Mat roi1(mask1, cv::Rect((int)-min_x, (int)-min_y, img1.cols, img1.rows));
    roi1.setTo(255);

    cv::Mat gray2, mask2;
    cv::cvtColor(warped2, gray2, cv::COLOR_BGR2GRAY);
    cv::threshold(gray2, mask2, 0, 255, cv::THRESH_BINARY);

    std::vector<cv::Mat> imgs = { warped1, warped2 };
    std::vector<cv::Mat> masks = { mask1, mask2 };
    std::vector<cv::Point> corners = { cv::Point(0,0), cv::Point(0,0) };

    // Convert to UMat (force types)
    std::vector<cv::UMat> imgs_um, masks_um;
    for (size_t i = 0; i < imgs.size(); ++i) {
        cv::UMat iu, mu;
        cv::Mat tmp;
        imgs[i].convertTo(tmp, CV_8UC3);
        tmp.copyTo(iu);
        masks[i].convertTo(mu, CV_8U);
        imgs_um.push_back(iu);
        masks_um.push_back(mu);
    }

    // Seam finder (Voronoi only)
    std::cout << "Finding seams (this may take a while)..." << std::endl;
    try {
        cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::VoronoiSeamFinder>();
        seam_finder->find(imgs_um, corners, masks_um);
        std::cout << "Seam finding completed successfully." << std::endl;
    }
    catch (cv::Exception& e) {
        std::cerr << "Voronoi failed: " << e.what() << "\nSkipping seam finding.\n";
    }
    catch (std::exception& e) {
        std::cerr << "Seam finding failed with error: " << e.what() << "\nSkipping seam finding.\n";
    }
    catch (...) {
        std::cerr << "Seam finding failed with unknown error. Skipping seam finding.\n";
    }

    // Copy masks back
    std::cout << "Copying masks back..." << std::endl;
    for (size_t i = 0; i < masks.size(); ++i) {
        masks_um[i].copyTo(masks[i]);
    }

    // Multi-band blending
    std::cout << "Blending images..." << std::endl;
    cv::detail::MultiBandBlender blender;
    blender.prepare(cv::Rect(0, 0, pano_w, pano_h));
    for (size_t i = 0; i < imgs.size(); ++i) {
        cv::Mat img_s, mask_s;
        imgs[i].convertTo(img_s, CV_16S);
        masks[i].copyTo(mask_s);
        blender.feed(img_s, mask_s, corners[i]);
    }

    cv::Mat result_s, result_mask;
    blender.blend(result_s, result_mask);
    cv::Mat result;
    result_s.convertTo(result, CV_8U);

    // Save & show
    std::string base_outname = image_left_name.substr(0, image_left_name.find_last_of('.')) +
        "_" + image_right_name.substr(0, image_right_name.find_last_of('.')) +
        "_det-" + detector_name;
    fs::path outpath = stored_folder / (base_outname + ".png");
    int counter = 1;
    while (fs::exists(outpath)) {
        outpath = stored_folder / (base_outname + "_" + std::to_string(counter) + ".png");
        counter++;
    }

    std::cout << "Attempting to save to: " << fs::absolute(outpath) << "\n";
    std::cout << "Directory exists: " << fs::exists(stored_folder) << "\n";
    
    if (cv::imwrite(path_to_string(outpath), result))
        std::cout << "Saved panorama at: " << fs::absolute(outpath) << "\n";
    else
        std::cerr << "Failed to save panorama at: " << fs::absolute(outpath) << "\n";

    // Save statistics image
    fs::path stats_path = stored_folder / (base_outname + "_stats.png");
    counter = 1;
    while (fs::exists(stats_path)) {
        stats_path = stored_folder / (base_outname + "_stats_" + std::to_string(counter) + ".png");
        counter++;
    }
    
    if (cv::imwrite(path_to_string(stats_path), stats_img))
        std::cout << "Saved statistics image at: " << fs::absolute(stats_path) << "\n";
    else
        std::cerr << "Failed to save statistics image at: " << fs::absolute(stats_path) << "\n";

    cv::imshow("Panorama", result);
    cv::imshow("Statistics & Feature Matching", stats_img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
