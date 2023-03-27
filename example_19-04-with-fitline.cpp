// Example 19-4. Two-dimensional line fitting
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;


void help(char **argv) {
  cout << "\nExample 19-04, two dimensional line fitting"
       << "\nCall"
       << "\n" << argv[0] << "\n"
       << "\n 'q', 'Q' or ESC to quit"
       << "\n" << endl;
}

// Copied from imgproc/src/linefit.cpp
// 向量法计算 dist 
// https://baike.baidu.com/item/%E7%82%B9%E5%88%B0%E7%9B%B4%E7%BA%BF%E8%B7%9D%E7%A6%BB/8673346
/*
static double calcDist2D(const Point2f* points, int count, float* _line, float* dist) {
    int j;
    float px = _line[2], py = _line[3];
    float nx = _line[1], ny = -_line[0];
    double sum_dist = 0.;

    for (j = 0; j < count; j++) {
        float x, y;

        x = points[j].x - px;
        y = points[j].y - py;

        dist[j] = (float)fabs(nx * x + ny * y);
        sum_dist += dist[j];
    }

    return sum_dist;
}
*/


static void fitLine2D_wods(const Point* points, int count, float* weights, float* line) {
    CV_Assert(count > 0);
    double x = 0, y = 0, x2 = 0, y2 = 0, xy = 0, w = 0;
    double dx2, dy2, dxy;
    int i;
    float t;

    // Calculating the average of x and y...
    if (weights == 0) {
        for (i = 0; i < count; i += 1) {
            x += points[i].x;
            y += points[i].y;
            x2 += points[i].x * points[i].x;
            y2 += points[i].y * points[i].y;
            xy += points[i].x * points[i].y;
        }
        w = (float)count;
    } else {
        for (i = 0; i < count; i += 1) {
            x += weights[i] * points[i].x;
            y += weights[i] * points[i].y;
            x2 += weights[i] * points[i].x * points[i].x;
            y2 += weights[i] * points[i].y * points[i].y;
            xy += weights[i] * points[i].x * points[i].y;
            w += weights[i];
        }
    }

    // 这两种计算斜率的方式，差别不大
    // https://juejin.cn/s/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95%E6%8B%9F%E5%90%88%E7%9B%B4%E7%BA%BF%E6%96%B9%E7%A8%8B
    // https://juejin.cn/post/7068904750296596511
    // https://juejin.cn/post/6844903475185188872
    t = (float) atan2(w * xy - x*y, w*x2 - x*x);

    x /= w;
    y /= w;
    x2 /= w;
    y2 /= w;
    xy /= w;

    dx2 = x2 - x * x;
    dy2 = y2 - y * y;
    dxy = xy - x * y;

    // t = (float)atan2(2 * dxy, dx2 - dy2) / 2;
    line[0] = (float)cos(t);
    line[1] = (float)sin(t);

    line[2] = (float)x;
    line[3] = (float)y;
}

int main(int argc, char* argv[]) {
	//创建一个用于绘制图像的空白图
	cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
 
	//输入拟合点
	std::vector<cv::Point> points;
	points.push_back(cv::Point(48, 58));
	points.push_back(cv::Point(105, 98));
	points.push_back(cv::Point(155, 160));
	points.push_back(cv::Point(212, 220));
	points.push_back(cv::Point(248, 260));
	points.push_back(cv::Point(320, 300));
	points.push_back(cv::Point(350, 360));
	points.push_back(cv::Point(412, 400));
 
	//将拟合点绘制到空白图上
	for (int i = 0; i < points.size(); i++)
	{
		cv::circle(image, points[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
	}
 
	cv::Vec4f line_para; 
	cv::fitLine(points, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
    //float line[4]={0.f};
    //fitLine2D_wods(points.data(), points.size(), 0, line);
    //line_para[0] = line[0];
    //line_para[1] = line[1];
    //line_para[2] = line[2];
    //line_para[3] = line[3];

	std::cout << "line_para = " << line_para << std::endl;
 
	//获取点斜式的点和斜率
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];
 
	double k = line_para[1] / line_para[0];
 
	//计算直线的端点(y = k(x - x0) + y0)
	cv::Point point1, point2;
	point1.x = 0;
	point1.y = k * (0 - point0.x) + point0.y;
	point2.x = 640;
	point2.y = k * (640 - point0.x) + point0.y;
 
	cv::line(image, point1, point2, cv::Scalar(0, 255, 0), 2, 8, 0);
 
	cv::imshow("image", image);
    cv::imwrite("fitline-origin.jpg", image);
	cv::waitKey(0);
	return 0;
}


int main2(int argc, char **argv) {
  cv::Mat img(500, 500, CV_8UC3);
  cv::RNG rng(-1);
  help(argv);
  for (;;) {
    char key;
    int i, count = rng.uniform(0, 100) + 3, outliers = count / 5;
    float a = (float)rng.uniform(0., 200.);
    float b = (float)rng.uniform(0., 40.);
    float angle = (float)rng.uniform(0., CV_PI);
    float cos_a = cos(angle), sin_a = sin(angle);
    cv::Point pt1, pt2;
    vector<cv::Point> points(count);
    float line[6]={0.f};
    float d, t;
    b = MIN(a * 0.3f, b);

    // generate some points that are close to the line
    for (i = 0; i < count - outliers; i++) {
      float x = (float)rng.uniform(-1., 1.) * a;
      float y = (float)rng.uniform(-1., 1.) * b;
      points[i].x = cvRound(x * cos_a - y * sin_a + img.cols / 2);
      points[i].y = cvRound(x * sin_a + y * cos_a + img.rows / 2);
    }

    // generate outlier points
    for (; i < count; i++) {
      points[i].x = rng.uniform(0, img.cols);
      points[i].y = rng.uniform(0, img.rows);
    }

    // find the optimal line
    //cv::fitLine(points, line, cv::DIST_L1, 1, 0.001, 0.001);
    // DIST_L2是最原始的最小二乘，最容易翻车的一种拟合方式，虽然速度快点。
    //cv::fitLine(points, line, cv::DIST_L2, 1, 0.001, 0.001);
    fitLine2D_wods(points.data(), points.size(), 0, line);

    // draw the points
    img = cv::Scalar::all(0);
    for (i = 0; i < count; i++)
      cv::circle(img, points[i], 2,
                 i < count - outliers ? cv::Scalar(0, 0, 255)
                                      : cv::Scalar(0, 255, 255),
                 cv::FILLED);

    // ... and the long enough line to cross the whole image
    d = sqrt((double)line[0] * line[0] + (double)line[1] * line[1]);
    line[0] /= d;
    line[1] /= d;
    t = (float)(img.cols + img.rows);
    pt1.x = cvRound(line[2] - line[0] * t);
    pt1.y = cvRound(line[3] - line[1] * t);
    pt2.x = cvRound(line[2] + line[0] * t);
    pt2.y = cvRound(line[3] + line[1] * t);
    cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0), 3);
    cv::imshow("Fit Line", img);
    key = (char)cv::waitKey(0);
    if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
      break;
  }
  return 0;
}
