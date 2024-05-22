#include <iostream>
#include "mpi.h"
#include "opencv2/opencv.hpp"

enum Operation { COLOR_SPACE_CONVERSION, GLOBAL_THRESHOLDING, LOCAL_THRESHOLDING, MEDIAN_FILTER, EDGE_DETECTION, GAUSSIAN_BLUR, IMAGE_ROTATION, IMAGE_SCALING };

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize selected operation variable
    Operation selectedOperation;
    double blurRadius = 0.0; // Initialize blur radius variable
    double rotationAngle = 0.0; // Initialize rotation angle variable
    double scalingFactor = 1.0; // Initialize scaling factor variable

    // Initialize MPI variables for timing
    double startTime, endTime;

    // Start timing
    startTime = MPI_Wtime();

    // Prompt the user to select the operation
    if (rank == 0)
    {
        std::cout << "Select the operation to apply:" << std::endl;
        std::cout << "0: Color Space Conversion to Grayscale" << std::endl;
        std::cout << "1: Global Thresholding" << std::endl;
        std::cout << "2: Local Thresholding" << std::endl;
        std::cout << "3: Median Filter" << std::endl;
        std::cout << "4: Edge Detection (Canny)" << std::endl;
        std::cout << "5: Gaussian Blur" << std::endl;
        std::cout << "6: Image Rotation" << std::endl;
        std::cout << "7: Image Scaling" << std::endl;

        int operation;
        std::cin >> operation;

        if (operation == COLOR_SPACE_CONVERSION)
            selectedOperation = COLOR_SPACE_CONVERSION;
        else if (operation == GLOBAL_THRESHOLDING)
            selectedOperation = GLOBAL_THRESHOLDING;
        else if (operation == LOCAL_THRESHOLDING)
            selectedOperation = LOCAL_THRESHOLDING;
        else if (operation == MEDIAN_FILTER)
            selectedOperation = MEDIAN_FILTER;
        else if (operation == EDGE_DETECTION)
            selectedOperation = EDGE_DETECTION;
        else if (operation == GAUSSIAN_BLUR)
            selectedOperation = GAUSSIAN_BLUR;
        else if (operation == IMAGE_ROTATION)
            selectedOperation = IMAGE_ROTATION;
        else if (operation == IMAGE_SCALING)
            selectedOperation = IMAGE_SCALING;
        else
        {
            std::cerr << "Invalid operation selected!" << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Broadcast selected operation to all processes
    MPI_Bcast(&selectedOperation, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast blur radius to all processes
    MPI_Bcast(&blurRadius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast rotation angle to all processes
    MPI_Bcast(&rotationAngle, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast scaling factor to all processes
    MPI_Bcast(&scalingFactor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Load image in the root process
    cv::Mat image;
    std::string inputImageName;
    if (rank == 0)
    {
        std::cout << "Enter the name of the input image: ";
        std::cin >> inputImageName;
        image = cv::imread(inputImageName, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "Image is empty, terminating!" << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Scatter the image data
    cv::Mat localImage;
    MPI_Bcast(&image.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    localImage = cv::Mat(image.rows, image.cols, CV_8UC3);
    MPI_Scatter(image.data, image.rows * image.cols * 3 / size, MPI_UNSIGNED_CHAR, localImage.data, image.rows * image.cols * 3 / size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Perform the selected operation
    cv::Mat resultImage;
    std::string outputImageName;
    if (selectedOperation == COLOR_SPACE_CONVERSION)
    {
        if (rank == 0)
        {
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        cv::cvtColor(localImage, resultImage, cv::COLOR_BGR2GRAY);
    }
    else if (selectedOperation == GLOBAL_THRESHOLDING)
    {
        if (rank == 0)
        {
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        cv::cvtColor(localImage, resultImage, cv::COLOR_BGR2GRAY);
        cv::threshold(resultImage, resultImage, 128, 255, cv::THRESH_BINARY);
    }
    else if (selectedOperation == LOCAL_THRESHOLDING)
    {
        if (rank == 0)
        {
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        cv::cvtColor(localImage, resultImage, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(resultImage, resultImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 2);
    }
    else if (selectedOperation == MEDIAN_FILTER)
    {
        if (rank == 0)
        {
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        cv::medianBlur(localImage, resultImage, 5);
    }
    else if (selectedOperation == EDGE_DETECTION)
    {
        if (rank == 0)
        {
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        cv::Mat grayImage;
        cv::cvtColor(localImage, grayImage, cv::COLOR_BGR2GRAY);
        cv::Canny(grayImage, resultImage, 100, 200);
    }
    else if (selectedOperation == GAUSSIAN_BLUR)
    {
        if (rank == 0)
        {
            std::cout << "Enter the blur radius: ";
            std::cin >> blurRadius;
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        MPI_Bcast(&blurRadius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Convert blurRadius to an odd integer (kernel size requirement)
        int kernelSize = static_cast<int>(2 * blurRadius) + 1;

        // Ensure kernel size is odd for Gaussian blur
        if (kernelSize % 2 == 0)
            kernelSize++;

        // Apply Gaussian blur
        cv::GaussianBlur(localImage, resultImage, cv::Size(kernelSize, kernelSize), blurRadius);
    }
    else if (selectedOperation == IMAGE_ROTATION)
    {
        if (rank == 0)
        {
            std::cout << "Enter the rotation angle (in degrees): ";
            std::cin >> rotationAngle;
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        MPI_Bcast(&rotationAngle, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform image rotation
        cv::Point2f center(localImage.cols / 2.0, localImage.rows / 2.0);
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotationAngle, 1.0);
        cv::warpAffine(localImage, resultImage, rotationMatrix, localImage.size());
    }
    else if (selectedOperation == IMAGE_SCALING)
    {
        if (rank == 0)
        {
            std::cout << "Enter the scaling factor: ";
            std::cin >> scalingFactor;
            std::cout << "Enter the name of the output image: ";
            std::cin >> outputImageName;
        }
        MPI_Bcast(&scalingFactor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform image scaling
        cv::resize(localImage, resultImage, cv::Size(), scalingFactor, scalingFactor);
    }

    // Gather the results
    cv::Mat globalResult;
    if (rank == 0)
    {
        globalResult = cv::Mat(image.rows, image.cols, CV_8UC1);
    }
    MPI_Gather(resultImage.data, image.rows * image.cols / size, MPI_UNSIGNED_CHAR, globalResult.data, image.rows * image.cols / size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // End timing
    endTime = MPI_Wtime();

    // Display or save the result image
    if (rank == 0)
    {
        cv::imwrite(outputImageName, globalResult);
        cv::imshow("Result Image", globalResult);
        cv::waitKey(0);

        // Print the execution time
        std::cout << "Execution time: " << endTime - startTime << " seconds" << std::endl;
        std::cout << "Thank You For Using Parallel Image Processing" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}