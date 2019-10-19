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
	Mat imgBinary;				//�������� �������� �����������
	Mat imageBinary;			//�������� �������� ����������� ���� CV_16UC3

	Mat model;					//����������� �������������� ������
	Mat modelShow;

	std::vector<infoWhitePixels> infoWhitePixelsVector;		//������ ���������� � ����� ��������

	std::vector<localMaximum> localMaximumsVector;
	std::vector<localMaximum> maximums;

	double alphaOrient(double);	//��������������� ���
	int round(double);			//����������

	int toX(int);
	int toY(int);

public:
	PHT();								//����������� �� ���������
	PHT(CannyClass, OperatorSobel);		//����������� � �������� ������ CannyClass � �������� ������ OperatorSobel

	Mat transformRadon(int, int, float, float, float, int);			//������������ �������� PHT � ������� Mat

	Mat showLines(int, int, int);		//����� �����

	Mat showLinesOnModel(int, int);		//����� ��������� ���������� �������������� ������
	Mat showFinalModel(Mat, int);		//��������� ������������� ������ �� �������� �����������

	Mat getModel();

};




#endif