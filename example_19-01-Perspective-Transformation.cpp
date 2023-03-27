// https://blog.csdn.net/guduruyu/article/details/72518340
#include <opencv2/opencv.hpp>
#include <iostream>



int main() {
    cv::Mat image = cv::imread("../20170519101748945.png");
	cv::Point2f src_points[] = { 
		cv::Point2f(165, 270),
		cv::Point2f(835, 270),
		cv::Point2f(360, 125),
		cv::Point2f(615, 125) };
    cv::circle(image, src_points[0], 9, cv::Scalar(255, 0, 0), 3);
    cv::circle(image, src_points[1], 9, cv::Scalar(0, 255, 0), 3);
    cv::circle(image, src_points[2], 9, cv::Scalar(0, 0, 255), 3);
    cv::circle(image, src_points[3], 9, cv::Scalar(0, 255, 255), 3);
    cv::imshow("Origin", image);
     
 
	cv::Point2f dst_points[] = {
		cv::Point2f(165, 270),
		cv::Point2f(835, 270),
		cv::Point2f(165, 30),
		cv::Point2f(835, 30) };
 
	cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);

 	cv::Mat perspective;
	cv::warpPerspective(image, perspective, M, cv::Size(960, 270), cv::INTER_LINEAR);
    std::cout << "Origin size:" << image.size() << ", after: " << perspective.size() << std::endl;

    cv::circle(perspective, dst_points[0], 9, cv::Scalar(255, 0, 0), 3);
    cv::circle(perspective, dst_points[1], 9, cv::Scalar(0, 255, 0), 3);
    cv::circle(perspective, dst_points[2], 9, cv::Scalar(0, 0, 255), 3);
    cv::circle(perspective, dst_points[3], 9, cv::Scalar(0, 255, 255), 3);

    cv::imshow("perspective", perspective);
    int key = cv::waitKey() & 255;
    //if (key == 27)
      //break;

    return 0;
}
