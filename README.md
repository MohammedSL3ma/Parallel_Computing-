# Parallel_Computing:
# Parallel Image Processing with MPI

This project demonstrates parallel image processing using MPI (Message Passing Interface) and OpenCV. The code distributes an image across multiple MPI processes, performs various image processing operations in parallel, and then gathers the results.

## Key Features

### Initialization
- *MPI Initialization*: Initializes MPI and determines the rank and size of MPI processes.
- *Input Validation*: Ensures correct command-line arguments are provided, with error messages for incorrect usage.

### Image Loading
- *Master Process*: The master process loads the image from the given path and validates the loaded image.

### Data Distribution
- *Broadcasting Dimensions*: Image dimensions (rows and columns) are broadcast to all MPI processes to maintain consistency.
- *Balanced Workload*: Calculates the number of rows each process will handle, considering extra rows for the last process to ensure balanced workload distribution.

### Local Image Processing
- *Processing Operations*: Each process applies specified image processing operations (e.g., Gaussian Blur, Edge Detection) to its local image chunk.

### Gathering Results
- *Collecting Processed Data*: Gathers processed image chunks from all MPI processes to reassemble the final image.

## Advantages
- *Enhanced Error Handling*: Robust error handling for command-line arguments and image loading ensures the program handles unexpected situations gracefully.
- *Balanced Workload Distribution*: Efficient utilization of resources across MPI processes leads to improved performance.
- *Consistent Data Distribution*: Broadcasting dimensions and scattering data ensure synchronized and coherent parallel processing.
- *Improved Code Structure*: Well-defined sections make the code easier to understand and maintain.

## Usage
```sh
mpiexec -n <number_of_processes> ./<executable> <image_path>
