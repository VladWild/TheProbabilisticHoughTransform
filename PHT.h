#pragma once
#ifndef PHT_H
#define PHT_H
#include "CannyClass.h"
#include "OperatorSobel.h"

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <search.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using cv::Mat;

struct point {
	int x;
	int y;
};

typedef struct {
	point location;
	float alpha;
	float alphaDirected;
	int s;
} infoWhitePixels;

typedef struct {
	float brightliness;
	point location;
} localMaximum;

class PHT
{
private:
	Mat imgBinary;				//исходное бинарное изображение
	Mat imageBinary;			//исходное бинарное изображение типа CV_16UC3

	Mat model;					//изображение преобразования Радона
	Mat modelShow;

	std::vector<infoWhitePixels> infoWhitePixelsVector;		//вектор информации о белых пикселях

	std::vector<localMaximum> localMaximumsVector;
	std::vector<localMaximum> maximums;

	double alphaOrient(double);	//ориентированный угл
	int round(double);			//округление

	int toX(int);
	int toY(int);

public:
	PHT();								//конструктор по умолчанию
	PHT(CannyClass, OperatorSobel);		//конструктор с объектом класса CannyClass и объектом класса OperatorSobel

	Mat transformRadon(int, int, float, float, float, int);			//формирование картинки PHT в формате Mat

	Mat showLines(int, int, int);		//показ линий

	Mat showLinesOnModel(int, int);		//поиск локальных максимумов преобразования Радона
	Mat showFinalModel(Mat, int);		//отрисовка прямолинейных границ на исходном изображении

	Mat getModel();

};




#endif