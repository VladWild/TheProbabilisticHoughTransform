#define _USE_MATH_DEFINES
#include "PHT.h"
#include "CannyClass.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <search.h>

using namespace cv;
using cv::Mat;

bool myfunction(localMaximum i, localMaximum j) {
	return (i.brightliness > j.brightliness);
}

//конструктор по умолчанию
PHT::PHT() : imgBinary(0, 0, CV_8UC1) {}

//конструктор с объектом класса CannyClass и объектом класса OperatorSobel
PHT::PHT(CannyClass operatorCanny, OperatorSobel operatorSobel)
{
	imgBinary = cvarrToMat(operatorCanny.getDst());
	imageBinary = imgBinary.clone();
	imageBinary.convertTo(imageBinary, CV_8UC1);

	/*for (int i = 10; i < 50; i++) {
		for (int j = 10; j < 50; j++) {
			std::cout << imageBinary.at<bool>(j, i) << " ";
		}
		std::cout << "\n";
	}*/

	//imshow("imageBinary", imageBinary);

	float alpha = 0;

	for (int i = 0; i < imageBinary.rows; i++) {
		for (int j = 0; j < imageBinary.cols; j++) {
			if (imageBinary.at<bool>(i, j)) {
				infoWhitePixels pixel;
				pixel.location.x = j;
				pixel.location.y = i;

				//--------------------------------------------------------------------

				//выясняем угл напрвления градиента alpha
				if (operatorSobel.getGrad_x().at<short int>(i, j) != 0) {		
					alpha = (double)atan((double)operatorSobel.getGrad_y().at<short int>(i, j) /
						(double)operatorSobel.getGrad_x().at<short int>(i, j));

					if ((double)operatorSobel.getGrad_x().at<short int>(i, j) < 0) {
						alpha += M_PI;
					}

					pixel.alpha = alphaOrient(alpha);
					
				}
				else {
					alpha = M_PI / 2;
					pixel.alpha = M_PI / 2;
				}

				//--------------------------------------------------------------------

				//выясняем направленный угл alphaDirected;
				double M_x0 = imageBinary.cols / 2;
				double M_y0 = imageBinary.rows / 2;

				int length_arrow = 10;

				//d_x0

				double d_x0 = sqrt(pow(j - M_x0, 2) + pow(i - M_y0, 2));

				//d_x10

				int x10 = j + length_arrow*cos(alpha);
				int y10 = i + length_arrow*sin(alpha);

				double d_x1 = sqrt(pow(x10 - M_x0, 2) + pow(y10 - M_y0, 2));

				if (d_x1 < d_x0) {
					continue;
					if (pixel.alpha > M_PI) {
						pixel.alphaDirected = pixel.alpha - M_PI;
					}
					else {
						pixel.alphaDirected = pixel.alpha + M_PI;
					}
				}
				else {
					pixel.alphaDirected = alpha;
				}


				//--------------------------------------------------------------------

				//выясняем расстояние от центра до прямой s
				float betta = alpha + M_PI / 2;
				float gamma = alpha + 3 * M_PI / 2;

				int r = sqrt(pow(imageBinary.cols, 2) + pow(imageBinary.rows, 2));

				int x2Current = r*cos(betta);
				int y2Current = r*sin(betta);

				int x4Current = r*cos(gamma);
				int y4Current = r*sin(gamma);

				double y1 = i + y2Current;
				double y2 = i + y4Current;
				double x1 = j + x2Current;
				double x2 = j + x4Current;

				double A = y1 - y2;
				double B = x2 - x1;
				double C = x1*y2 - x2*y1;

				double M_x = imageBinary.cols / 2;
				double M_y = imageBinary.rows / 2;

				pixel.s = round(abs(A*M_x + B*M_y + C) / sqrt(A*A + B*B));
				//--------------------------------------------------------------------

				//std::cout << "x=" << pixel.location.x << " y=" << pixel.location.y << ":  "
				//	<< "grad_y= " << operatorSobel.getGrad_y().at<short int>(i, j) << " " 
				//	<< "grad_x= " << operatorSobel.getGrad_x().at<short int>(i, j) << " "
				//	<< " alpha= " << alpha
				//	<< " pixel.alpha= " << pixel.alpha
				//	<< " s= " << pixel.s << "\n";

				infoWhitePixelsVector.push_back(pixel);
			}
		}
	}

	std::cout << "NumberWhitePixels: " << infoWhitePixelsVector.size() << "\n";

}

//округление
int PHT::round(double number) {
	return (int) (number + 0.5);
}

//ориентированный угл
double PHT::alphaOrient(double alpha) {
	if (alpha < 0) {
		return abs(alpha);
	}
	else {
		return 2 * M_PI - alpha;
	}
}

//формирование картинки PHT в формате Mat
Mat PHT::transformRadon(int dA, int dS, float dispA, float dispS, float p, int numberOfAngles) {

	std::vector<infoWhitePixels>::iterator it = infoWhitePixelsVector.begin();
	int maxS = 0;

	for (it = infoWhitePixelsVector.begin(); it != infoWhitePixelsVector.end(); ++it) {
		if ((*it).s > maxS) {
			maxS = (*it).s;
		}
	}

	Mat model(maxS + 1, numberOfAngles + 1, CV_16UC1, Scalar::all(0));
	Mat modelFloat(maxS + 1, numberOfAngles + 1, CV_32FC1, Scalar::all(0));

	infoWhitePixels pixel;
	int numberAlpha;
	double stepofAngles = 2 * M_PI / numberOfAngles;

	for (it = infoWhitePixelsVector.begin(); it != infoWhitePixelsVector.end(); ++it) {

		pixel = (*it);
		numberAlpha = trunc(pixel.alpha / stepofAngles);

		for (int i = -dS; i <= dS; i++) {
			for (int j = -dA; j <= dA; j++) {
				 if ((numberAlpha + j > 0) &&
					(numberAlpha + j < model.cols) &&
					(pixel.s + i > 0) &&
					(pixel.s + i < model.rows))
					{
						modelFloat.at<float>(pixel.s + i, numberAlpha + j) += (1 - p) / (4 * dS * dA) +
							 p / (2 * M_PI * sqrt(dispS) * sqrt(dispA)) *
							 exp(-pow(i, 2) / (2 * dispS) - pow(j, 2) / (2 * dispA));
					}
			}
		}
		
	}
	
	(*this).model.convertTo(model, CV_32FC1);
	(*this).model = modelFloat.clone();

	std::cout << "The construction of the model is complete\n\n";

	return (*this).model;
}

//показ линий
Mat PHT::showLines(int alpha, int s, int numberOfAngles) {

	std::cout << "alpha = " << alpha << "\n s = " << s << "   \n\n";
	Mat show = imgBinary.clone();
	Scalar scalar = Scalar::all(156);
	line(show, Point(show.cols / 2, 0), Point(show.cols / 2, show.rows), scalar);
	line(show, Point(0, show.rows / 2), Point(show.cols, show.rows / 2), scalar);
	line(show, Point(show.cols / 2, 0), Point(show.cols / 2 - 5, 10), scalar);
	line(show, Point(show.cols / 2, 0), Point(show.cols / 2 + 5, 10), scalar);
	line(show, Point(show.cols, show.rows / 2), Point(show.cols - 10, show.rows / 2 - 5), scalar);
	line(show, Point(show.cols, show.rows / 2), Point(show.cols - 10, show.rows / 2 + 5), scalar);
	putText(show, "x", Point(show.cols - 24, show.cols / 2 - 24), FONT_HERSHEY_COMPLEX, 0.5, scalar, 1, 8);
	putText(show, "y", Point(show.cols / 2 + 24, 16), FONT_HERSHEY_COMPLEX, 0.5, scalar, 1, 8);

	float alpha_rad = 2 * alpha * M_PI / numberOfAngles;
	int xCurrent = s*cos(alpha_rad);
	int yCurrent = s*sin(alpha_rad);
	int x = toX(xCurrent);
	int y = toY(yCurrent);
	line(show, Point(show.cols / 2, show.rows / 2), Point(x, y), scalar);

	float betta = alpha_rad + M_PI / 2;
	float gamma = alpha_rad + 3 * M_PI / 2;

	int r = sqrt(pow(show.cols, 2) + pow(show.rows, 2));

	int x2Current = xCurrent + r*cos(betta);
	int y2Current = yCurrent + r*sin(betta);
	int x2 = toX(x2Current);
	int y2 = toY(y2Current);
	line(show, Point(x, y), Point(x2, y2), scalar);

	int x4Current = xCurrent + r*cos(gamma);
	int y4Current = yCurrent + r*sin(gamma);
	int x4 = toX(x4Current);
	int y4 = toY(y4Current);
	line(show, Point(x, y), Point(x4, y4), scalar);

	return show;
}

//поиск локальных максимумов преобразования Радона
Mat PHT::showLinesOnModel(int numberOfLines, int maximumDistance)
{
	maximums.clear();
	modelShow = model.clone();

	localMaximum localMaximumElement;
	for (int i = 0; i < (*this).modelShow.rows; i++) {
		for (int j = 0; j < (*this).modelShow.cols; j++) {
			localMaximumElement.brightliness = (*this).modelShow.at<float>(i, j);
			localMaximumElement.location.x = j;
			localMaximumElement.location.y = i;
			localMaximumsVector.push_back(localMaximumElement);
		}
	}

	std::sort(localMaximumsVector.begin(), localMaximumsVector.end(), myfunction);

	std::vector<localMaximum>::iterator it = localMaximumsVector.begin();
	localMaximum lastElement;
	bool pixel = false;
	maximums.push_back(*it);

	for (it = localMaximumsVector.begin(); it != localMaximumsVector.end(); ++it) {
		lastElement = *it;
		for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
			if ((abs((*it_max).location.x - lastElement.location.x) <= maximumDistance) ||
				(abs((*it_max).location.y - lastElement.location.y) <= maximumDistance)) {
				pixel = false;
				break;
			}
			else {
				pixel = true;
				continue;
			}
		};
		if (pixel == true) {
			maximums.push_back(lastElement);
			if (maximums.size() == numberOfLines) {
				break;
			}
			else {
				continue;
			}
		}
		else {
			continue;
		}
	};

	for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
		localMaximum k = *it_max;
		std::cout << "brightliness = " << k.brightliness
			<< " x=" << k.location.x
			<< " y=" << k.location.y
			<< "\n";
	};

	Mat modified = (*this).modelShow.clone();

	for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
		int x1 = (*it_max).location.x - maximumDistance;
		int y1 = (*it_max).location.y - maximumDistance;
		if (x1 <= 0) x1 = 0;
		if (y1 <= 0) y1 = 0;
		Point pt1(x1, y1);
		int x2 = (*it_max).location.x + maximumDistance;
		int y2 = (*it_max).location.y + maximumDistance;
		if (x2 >= modified.cols) x2 = modified.cols;
		if (y2 >= modified.rows) y2 = modified.rows;
		Point pt2(x2, y2);
		rectangle(modified, pt1, pt2, Scalar::all(65536), 1, 8, 0);
	};

	return modified;
}

//отрисовка прямолинейных границ на исходном изображении
Mat PHT::showFinalModel(Mat modelCurrent, int numberOfAngles) {
	for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
		int alpha = (*it_max).location.x;
		int s = (*it_max).location.y;

		std::cout << "alpha = " << alpha << "\n s = " << s << "   \n\n";

		//Scalar scalar = Scalar::all(255);
		Scalar scalar = Scalar::all(0);

		float alpha_rad = 2 * alpha * M_PI / numberOfAngles;
		int xCurrent = s*cos(alpha_rad);
		int yCurrent = s*sin(alpha_rad);
		int x = toX(xCurrent);
		int y = toY(yCurrent);

		float betta = alpha_rad + M_PI / 2;
		float gamma = alpha_rad + 3 * M_PI / 2;

		int r = sqrt(pow(modelCurrent.cols, 2) + pow(modelCurrent.rows, 2));

		int x2Current = xCurrent + r*cos(betta);
		int y2Current = yCurrent + r*sin(betta);
		int x2 = toX(x2Current);
		int y2 = toY(y2Current);
		line(modelCurrent, Point(x, y), Point(x2, y2), scalar, 2);

		int x4Current = xCurrent + r*cos(gamma);
		int y4Current = yCurrent + r*sin(gamma);
		int x4 = toX(x4Current);
		int y4 = toY(y4Current);
		line(modelCurrent, Point(x, y), Point(x4, y4), scalar, 2);
	};

	return modelCurrent;
}

//преобразование координаты x
int PHT::toX(int x) {
	return x + imageBinary.cols / 2;
}

//преобразование координаты y
int PHT::toY(int y) {
	return imageBinary.rows / 2 - y;
}

//получение модели
Mat PHT::getModel() {
	return model;
}

