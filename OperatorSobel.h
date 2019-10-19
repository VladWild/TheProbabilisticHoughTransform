#pragma once
#ifndef OPERATORSOBEL_H
#define OPERATORSOBEL_H

#include <opencv2/opencv.hpp>

using namespace cv;

class OperatorSobel
{
private:
	const int scale = 1;
	const int delta = 0;
	const int ddepth = CV_16S;

	std::string stringFileName;

	Mat src, src_gray;				//исходное изображение и его черно-белая версия
	Mat grad_x, grad_y;				//градиенты по x и по у в формате CV_16S
	Mat abs_grad_x, abs_grad_y;		//модули градиентов по x и по у
	Mat grad;						//слияние градиентов (апроксимация)

public:
	OperatorSobel();
	OperatorSobel(std::string);

	void setOperatorSobel(std::string);

	Mat getScr();
	Mat getSrc_gray();
	Mat getGrad_x();
	Mat getGrad_y();
	Mat getAbs_grad_x();
	Mat getAbs_grad_y();
	Mat getGrad();

};

#endif OPERATORSOBEL_H