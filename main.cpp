// https://docs.opencv.org/master/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

using namespace std::literals;
using std::size_t;

class TrackbarVariable
{
    inline static std::string windowName;

public:
    static void setWindowName(std::string windowName_in)
    {
        windowName = std::move(windowName_in);
    }

private:
    int variable;

public:
    TrackbarVariable(std::string name, int max, int initial = 0)
        : variable(initial)
    {
        cv::createTrackbar(name, windowName, &variable, max); // I think 0 is an error return value? Nothing in the docs
    }

    TrackbarVariable& operator=(int rhs)
    {
        variable = rhs;

        return *this;
    }

    operator int&()
    {
        return variable;
    }
};

const std::string
    trackbarWindowName("Trackbars"),
    videoWindowName("video"),
    x_componentWindowName("X velocity"),
    x_thresholdedWindowName("X velocity thresholded");

void spawnWindows()
{
#ifdef WIN32
    SetConsoleTitle("Average X velocity");
    MoveWindow(GetConsoleWindow(), 0, 0, 300, 1000, false);
#endif

    cv::namedWindow(videoWindowName);
    cv::namedWindow(x_componentWindowName);
    cv::namedWindow(x_thresholdedWindowName);
    cv::namedWindow(trackbarWindowName);

    cv::moveWindow(videoWindowName, 300, 0);
    cv::moveWindow(x_componentWindowName, 1200, 0);
    cv::moveWindow(x_thresholdedWindowName, 300, 520);
    cv::moveWindow(trackbarWindowName, 1200, 520);

    cv::resizeWindow(trackbarWindowName, 300, 120);
}

void pushFrame(std::deque<cv::Mat>& frames, cv::Mat& original, int frames_n)
{
    // Flip the video so that the webcam is like a mirror
    cv::Mat frame;
    cv::flip(original, frame, 1);
    original = frame;

    // Lose colour information and convert to double representation to avoid range issues
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_64F);

    frames.push_front(frame.clone());

    // Discard old frames
    if (std::size(frames) < frames_n)
        return;
    
    frames.erase(std::begin(frames) + frames_n, std::end(frames));
}

std::vector<cv::Point> getChunkPositions(cv::Size frameSize, int chunkSize)
{
    std::vector<cv::Point> positions;
    for (int y = chunkSize / 2; y < frameSize.height - chunkSize / 2; y += chunkSize)
        for (int x = chunkSize / 2; x < frameSize.width - chunkSize / 2; x += chunkSize)
            positions.push_back({x, y});

    return positions;
}

void remove_contours(std::vector<std::vector<cv::Point>>& contours, double cmin)
{
    auto it = std::remove_if(std::begin(contours), std::end(contours), [=](const std::vector<cv::Point>& data)
    {
        return cmin <= contourArea(data);
    });
    contours.erase(it, std::end(contours));
}

void getFrameDerivatives(const cv::Mat& input, cv::Mat& partial_x, cv::Mat& partial_y)
{
    const double partial_array[] = {0.5, -0.5};

    cv::Mat partial_x_matrix(cv::Size(2, 1), CV_64F, (void*)partial_array);
    cv::Mat partial_y_matrix(cv::Size(1, 2), CV_64F, (void*)partial_array);

    cv::filter2D(input, partial_x, -1, partial_x_matrix);
    cv::filter2D(input, partial_y, -1, partial_y_matrix);
}

auto getDerivatives(std::deque<cv::Mat>& frames) -> std::tuple<cv::Mat, cv::Mat, cv::Mat>
{
    // Get spatial and temporal differences, averaged over last `size(frames)` frames //

    static TrackbarVariable average_space("Neighbourhood size", 20, 0); // Number of pixels above/below/left/right of working pixel to average over

    // Return variables
    cv::Mat partialX, partialY, partialT;

    const cv::Mat zeros = cv::Mat::zeros(std::size(frames[0]), frames[0].type());
    partialX = zeros.clone();
    partialY = zeros.clone();

    // Time derivative
    std::deque<cv::Mat> differences(std::size(frames));
    std::adjacent_difference(std::begin(frames), std::end(frames), std::begin(differences));
    differences[0] = zeros;
    partialT = std::accumulate(std::begin(differences), std::end(differences), zeros) / double(std::size(frames));

    // Calculate spatial derivatives for each frame
    std::vector<cv::Mat> partials_x, partials_y;
    for (cv::Mat& frame : frames)
    {
        cv::Mat partial_x_temp, partial_y_temp;
        getFrameDerivatives(frame, partial_x_temp, partial_y_temp);
        partials_x.push_back(std::move(partial_x_temp));
        partials_y.push_back(std::move(partial_y_temp));
    }

    // Average spatial derivatives over 'average_space' neighbouring pixels
    for (cv::Mat& partial_x : partials_x)
    {
        cv::copyMakeBorder(partial_x, partial_x, average_space / 2, average_space - average_space / 2, 0, 0, cv::BORDER_REPLICATE);
        for (int i{}; i <= average_space; ++i)
            partialX += partial_x.rowRange(i, partial_x.rows - average_space + i);
    }
    
    for (cv::Mat& partial_y : partials_y)
    {
        cv::copyMakeBorder(partial_y, partial_y, 0, 0, average_space / 2, average_space - average_space / 2, cv::BORDER_REPLICATE);
        for (int i{}; i <= average_space; ++i)
            partialY += partial_y.colRange(i, partial_y.cols - average_space + i);
    }
    
    partialX /= double(std::size(frames) * (average_space + 1));
    partialY /= double(std::size(frames) * (average_space + 1));

    return {partialX, partialY, partialT};
}

std::vector<double> maxMaxNormalise(const std::vector<double>& data)
{
    static double
        global_max{},
        global_min{};

    const double
        local_max(*std::max_element(std::begin(data), std::end(data))),
        local_min(*std::min_element(std::begin(data), std::end(data)));

    global_max = std::max(global_max, local_max);
    global_min = std::min(global_min, local_min);

    std::vector<double> normalised(data);
    for (double& v : normalised)
        v = (v - global_min) / (global_max - global_min);

    return normalised;
}

std::pair<std::vector<double>, std::vector<double>> LKTracker(const std::vector<cv::Point> positions, int size, const cv::Mat_<double>& partialX, const cv::Mat_<double>& partialY, const cv::Mat_<double>& partialT)
{
    std::vector<double> x_velocity, y_velocity;

    for (cv::Point2i position : positions)
    {
        cv::Matx<double, 2, 2> A(0, 0, 0, 0);
        cv::Vec2d B(0, 0);
        const int
            y_min = std::max(0, position.y - size),
            y_max = std::min(position.y + size, partialX.rows),
            x_min = std::max(0, position.x - size),
            x_max = std::min(position.x + size, partialX.cols);
        
        for (int y(y_min); y < y_max; ++y)
            for (int x(x_min); x < x_max; ++x)
            {
                A(0, 0) += partialX[y][x] * partialX[y][x];
                A(0, 1) += partialX[y][x] * partialY[y][x];
                A(1, 1) += partialY[y][x] * partialY[y][x];
                B[0] += partialT[y][x] * partialX[y][x];
                B[1] += partialT[y][x] * partialY[y][x];
            }
        
        A(1, 0) = A(0, 1);
        cv::Vec2d temp = A.inv() * B;
        x_velocity.push_back(temp[0]);
        y_velocity.push_back(temp[1]);
    }

    return {x_velocity, y_velocity};
}

std::string getGestureStatus(double total_x, size_t total_cells_x)
{
    static TrackbarVariable
        total_threshold("Speed threshold", 60, 10),
        time_threshold("Gesture time", 10, 0); // Number of frames of motion required to detect gesture

    static int time(0);
    static enum { left, right, none } status(none);
    if (total_cells_x == 0)
    {
        status = none;
        time = 0;
    }
    else
    {
        // Write average velocity to console
        total_x /= double(total_cells_x);
        std::cout << int(total_x) << std::endl;

        if (total_x > total_threshold)
        {
            if (status == left)
                ++time;
            else if (status == right)
                time = 0;

            status = left;
        }
        else if (total_x < -total_threshold)
        {
            if (status == right)
                ++time;
            else if (status == left)
                time = 0;

            status = right;
        }
        else
        {
            status = none;
            // time = 0;
        }
    }

    std::string status_string;
    if (status == none)
        status_string = "none"s;
    else if (time < time_threshold)
        status_string = "none - "s + std::to_string(time_threshold - time);
    else if (status == left)
        status_string = "left"s;
    else
        status_string = "right"s;

    return status_string;
}

int main()
{
    cv::VideoCapture video_in;
    video_in.setExceptionMode(true);
    try
    {
        video_in.open(0);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Could not capture video from camera: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    spawnWindows();

    // Create trackbars for various constants
    TrackbarVariable::setWindowName(trackbarWindowName);
    TrackbarVariable
        chunkSize("Chunk size", 100, 20),
        frames_n("Temporal blur", 8, 2),
        magnitude_threshold("Magnitude threshold", 100, 3),
        angle_threshold("Angle threshold", 90, 45), // Degrees
        size_min("Contour size", 15000, 10000), // Minimum contour size
        smoothing_size("Smoothing size", 5, 0);

    cv::Mat original;
    std::deque<cv::Mat> frames; // A circular queue of frames used in processing
    std::vector<cv::Point> chunkPositions;

    // Generate grid of blocks to LK over
    int lastChunkSize = 0;
    for (;;)
    {
        // Stream video input
        video_in >> original;
        if (original.empty())
        {
            std::clog << "Camera stream ended\n";
            break;
        }

        pushFrame(frames, original, frames_n);

        // Wait until enough frames have been received before processing
        if (std::size(frames) < frames_n)
            continue;
        
        // Possibly reinitialise chunk co-ordinates
        if (chunkSize != lastChunkSize)
        {
            chunkSize = std::max<int>(1, chunkSize);
            lastChunkSize = chunkSize;
            chunkPositions = getChunkPositions(original.size(), chunkSize);
        }

        // Calculate spatial and temporal partial derivatives for all pixels
        cv::Mat partialX, partialY, partialT;
        std::tie(partialX, partialY, partialT) = getDerivatives(frames);

        // Perform LK to estimate velocities of each block
        std::vector<double> x_velocity, y_velocity;
        std::tie(x_velocity, y_velocity) = LKTracker(chunkPositions, chunkSize, partialX, partialY, partialT);

        // Normalise x velocity for display purposes
        std::vector<double> x_velocity_normalised(maxMaxNormalise(x_velocity));

        // The images of the x components of the LK with zero as neutral grey
        cv::Mat_<double> x_component(original.size(), 0.5);
        cv::Mat_<double> x_thresholded(original.size(), 0.5);
        cv::Mat_<uchar>
            left_thresholded(original.size(), 0),
            right_thresholded(original.size(), 0),
            up_thresholded(original.size(), 0),
            down_thresholded(original.size(), 0);

        // Calculate average velocity of blocks subject to thesholding
        double total_x{};
        double total_y{};
        size_t total_cells_x{};
        size_t total_cells_y{};
#pragma omp parallel for reduction(+: total_x, total_y, total_cells_x, total_cells_y)
        for (size_t i = 0; i < std::size(chunkPositions); ++i)
        {
            const cv::Rect chunkRegion(chunkPositions[i].x - chunkSize / 2, chunkPositions[i].y - chunkSize / 2, chunkSize, chunkSize);

            // Draw raw x velocity for each block
            cv::rectangle(x_component, chunkRegion, x_velocity_normalised[i], cv::FILLED);

            // Magnitude threshold
            if (std::hypot(x_velocity[i], y_velocity[i]) < magnitude_threshold)
                continue;

            // Draw velocity vector
            cv::line(original, chunkPositions[i], chunkPositions[i] + cv::Point(int(x_velocity[i]), int(y_velocity[i])), cv::Scalar(255, 0, 0), 2);
            cv::circle(original, chunkPositions[i], 1, cv::Scalar(0, 255, 0), 1);

            // Angle threshold
            if (cv::fastAtan2(float(std::abs(y_velocity[i])), float(std::abs(x_velocity[i]))) < angle_threshold)
            {
                // Draw binary thresholded velocity
                cv::rectangle(x_thresholded, chunkRegion, x_velocity[i] > 0, cv::FILLED);

                // Add to total x velocity
                total_x += x_velocity[i];
                ++total_cells_x;

                if (x_velocity[i] > 0)
                    cv::rectangle(left_thresholded, chunkRegion, 255, cv::FILLED);
                else
                    cv::rectangle(right_thresholded, chunkRegion, 255, cv::FILLED);
            }

            if (cv::fastAtan2(float(std::abs(x_velocity[i])), float(std::abs(y_velocity[i]))) < angle_threshold)
            {
                // Add to total y velocity
                total_y += y_velocity[i];
                ++total_cells_y;

                if (y_velocity[i] > 0)
                    cv::rectangle(up_thresholded, chunkRegion, 255, cv::FILLED);
                else
                    cv::rectangle(down_thresholded, chunkRegion, 255, cv::FILLED);
            }
        }

        // Find contours
        if (smoothing_size != 0)
        {
            const int smooth_chunk_size = chunkSize * (smoothing_size * 2 - 1);
            const cv::Size kernel_size(smooth_chunk_size, smooth_chunk_size);

            cv::blur(left_thresholded, left_thresholded, kernel_size);
            cv::blur(right_thresholded, right_thresholded, kernel_size);
            cv::blur(up_thresholded, up_thresholded, kernel_size);
            cv::blur(down_thresholded, down_thresholded, kernel_size);
        }

        std::vector<std::vector<cv::Point>> left_contours, right_contours, up_contours, down_contours;

        cv::findContours(left_thresholded, left_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(right_thresholded, right_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(up_thresholded, up_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(down_thresholded, down_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        remove_contours(left_contours, size_min);
        remove_contours(right_contours, size_min);
        remove_contours(up_contours, size_min);
        remove_contours(down_contours, size_min);

        cv::drawContours(original, left_contours, -1, cv::Scalar(0, 0, 255), 5, 8);
        cv::drawContours(original, right_contours, -1, cv::Scalar(0, 255, 0), 5, 8);
        cv::drawContours(original, up_contours, -1, cv::Scalar(0, 255, 255), 5, 8);
        cv::drawContours(original, down_contours, -1, cv::Scalar(255, 255, 0), 5, 8);

        // Display detected gesture
        const std::string status_string(getGestureStatus(total_x, total_cells_x));
        cv::putText(x_thresholded, status_string, {50, 50}, cv::FONT_HERSHEY_SIMPLEX, 1, 1);

        cv::imshow(videoWindowName, original);
        cv::imshow(x_componentWindowName, x_component);
        cv::imshow(x_thresholdedWindowName, x_thresholded);

        // waitKey must follow imshow
        cv::waitKey(20); // 50 Hz
    }
}