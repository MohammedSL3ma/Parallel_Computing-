#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "mpi.h"
#include "opencv2/opencv.hpp"

// Define constants for algorithms
const int GAUSSIAN_BLUR = 0;
const int EDGE_DETECTION = 1;
const int IMAGE_ROTATION = 2;
const int IMAGE_SCALING = 3;
const int HISTOGRAM_EQUALIZATION = 4;

// Function prototypes
void applyAlgorithm(cv::Mat& image, int algorithm);
void displayImage(const cv::Mat& image, const std::string& windowName);
void performParallelProcessing(cv::Mat& image, int algorithm, int rank);

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Load image on master node (rank 0)
    cv::Mat image;
    if (rank == 0) {
        image = cv::imread("C:/Users/El-Wattaneya/Pictures/Camera Roll/1.jpg", cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image. Exiting..." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image size to all processes
    int imageRows = image.rows;
    int imageCols = image.cols;
    MPI_Bcast(&imageRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate partial size (number of pixels) for each process
    int partialSize = (imageRows * imageCols) / size;
    std::vector<uchar> partialBuffer(partialSize * 3); // 3 channels (BGR) per pixel

    // Scatter image data across processes
    MPI_Scatter(image.data, partialSize * 3, MPI_UNSIGNED_CHAR, partialBuffer.data(), partialSize * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // User enters the filter to apply
    int selectedFilter;
    if (rank == 0) {
        std::cout << "Enter the filter you want to apply:" << std::endl;
        std::cout << "0: Gaussian Blur" << std::endl;
        std::cout << "1: Edge Detection" << std::endl;
        // Add more filter options here
        std::cin >> selectedFilter;
    }

    // Broadcast selected filter to all processes
    MPI_Bcast(&selectedFilter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform parallel processing
    performParallelProcessing(partialBuffer, selectedFilter, rank, imageRows, imageCols);

    // Gather processed data back to master
    std::vector<uchar> processedData(imageRows * imageCols * 3);
    MPI_Gather(partialBuffer.data(), partialSize * 3, MPI_UNSIGNED_CHAR, processedData.data(), partialSize * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Display or save processed image on master
    if (rank == 0) {
        cv::Mat processedImage(imageRows, imageCols, CV_8UC3, processedData.data());
        displayImage(processedImage, "Processed Image");
        cv::imwrite("C:/Users/El-Wattaneya/Pictures/Camera Roll/1.jpg", processedImage);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

// Function to perform parallel processing based on selected algorithm
void performParallelProcessing(std::vector<uchar>& partialBuffer, int algorithm, int rank, int imageRows, int imageCols) {
    // Apply selected algorithm
    cv::Mat partialImage(imageRows, imageCols, CV_8UC3, partialBuffer.data()); // Reconstructing partial image
    switch (algorithm) {
    case GAUSSIAN_BLUR:
        applyGaussianBlur(partialImage);
        break;
    case EDGE_DETECTION:
        applyEdgeDetection(partialImage);
        break;
        // Add cases for other algorithms
    default:
        std::cerr << "Invalid algorithm choice" << std::endl;
    }
}


// Function to apply Gaussian blur algorithm
void applyGaussianBlur(cv::Mat& image) {
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);
}

// Function to apply edge detection algorithm
void applyEdgeDetection(cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Canny(grayImage, image, 50, 150);
}

// Function to display image
void displayImage(const cv::Mat& image, const std::string& windowName) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, image);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}
