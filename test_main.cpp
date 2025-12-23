#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>   // C++17 feature

namespace fs = std::filesystem;

int main() {
    // Folder containing your images
    fs::path image_folder = R"(C:\Users\madsm\source\repos\Visual_Computing_Nilsson\images)";
    std::string image_name = "ship_1.jpg";
    fs::path image_path = image_folder / image_name;

    std::cout << "Trying to load image from: " << image_path << std::endl;

    // Load the image
    cv::Mat img = cv::imread(image_path.string());

    if (img.empty()) {
        std::cerr << "Could not read the image from: " << image_path << std::endl;
        return 1;
    }

    // Display the image
    cv::imshow("Display window", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Folder to store copied images
    fs::path stored_folder = image_folder / "stored_images";

    // Create folder if it does not exist
    if (!fs::exists(stored_folder)) {
        if (!fs::create_directory(stored_folder)) {
            std::cerr << "Failed to create folder: " << stored_folder << std::endl;
            return 1;
        }
    }

    // Path to save the copied image
    fs::path stored_image_path = stored_folder / image_name;

    // Save the image
    if (!cv::imwrite(stored_image_path.string(), img)) {
        std::cerr << "Failed to save image to: " << stored_image_path << std::endl;
        return 1;
    }

    std::cout << "Image saved successfully to: " << stored_image_path << std::endl;

    return 0;
}
