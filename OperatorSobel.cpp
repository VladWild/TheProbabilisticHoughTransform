#include "OperatorSobel.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using cv::Mat;

OperatorSobel::OperatorSobel() {}

OperatorSobel::OperatorSobel(std::string stringFileName) {
	(*this).stringFileName = stringFileName;
	src = imread((*this).stringFileName);
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(src, src_gray, CV_RGB2GRAY);										//конвертируем его
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);		//градиент по x
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);		//градиент по y
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);						//слияние градиентов по x и по y

}

void OperatorSobel::setOperatorSobel(std::string stringFileName) {
	(*this).stringFileName = stringFileName;
}

Mat OperatorSobel::getScr() {
	return src;
}

Mat OperatorSobel::getSrc_gray() {
	return src_gray;
}

Mat OperatorSobel::getGrad_x() {
	return grad_x;
}

Mat OperatorSobel::getGrad_y() {
	return grad_y;
}

Mat OperatorSobel::getAbs_grad_x() {
	return abs_grad_x;
}

Mat OperatorSobel::getAbs_grad_y() {
	return abs_grad_y;
}

Mat OperatorSobel::getGrad() {
	return grad;
}




