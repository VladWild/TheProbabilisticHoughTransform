#include "Menu.h"
#include "CannyClass.h"
#include "OperatorSobel.h"
#include "PHT.h"
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

using namespace cv;
using cv::Mat;

//название окон
std::string cvNameCanny = "cvCanny";
std::string nameOriginalImage = "OriginalImage";
std::string nameRadon = "Radon";

//название трекбаров
std::string cvNameCannyTrackbar = "cvCannyTrackbar";
std::string nameRadonTrackbar = "RadonTrackbar";
std::string nameSearchLines = "SearchLinesTrackbar";

//им€ картинки
//std::string filename = "20.jpg";
//std::string filename = "21.jpg";
//std::string filename = "53385.jpg";
//std::string filename = "24.jpg";
//std::string filename = "7405.jpg";
//std::string filename = "2.jpg";
//std::string filename = "4.jpg";
//std::string filename = "1doroga.jpg";
//std::string filename = "24c154085fda707.jpg";
//std::string filename = "xYASbDNCudo.jpg";
//std::string filename = "sdvsrvdrvdrv.jpg";
//std::string filename = "42.jpg";
//std::string filename = "56.jpg";
std::string filename = "0_39d17_e937dce4_L.jpg";
//std::string filename = "5555.png";
//std::string filename = "5235abf383095.jpg";
//std::string filename = "10975cp6u5Wx0E8hImlCq4jH4r.jpg";
//std::string filename = "tmb_184199_9283.jpg";
//std::string filename = "polka4.jpg";
//std::string filename = "malogabaritnyy_kompyuternyy_stol_4 (1).jpg";
//std::string filename = "29de09d7a42d5742edfeac918b01ddb6.jpg";


//им€ файла, в который записываетс€ Mat
const char* filenametxt = "ImagePHT.txt";

//запись Mat в файл
void writeMatToFile(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);
	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;
		return;
	}
	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<float>(i, j) << " ";
		}
		fout << std::endl;
	}
	fout.close();
}

//глобальные переменые объектов собственных классов
CannyClass operatorCanny(filename);
OperatorSobel operatorSobel(filename);
PHT transformRadon;

//глобальные переменые дл€ трекбаров
const int cannyTreshhold1Max = 1000;
int cannyTreshhold1 = 100;
const int cannyTreshhold2Max = 1000;
int cannyTreshhold2 = 100;
const int operatorSobelMax = 1;
int operatorSobel_tb = 0;
const int dAMax = 16;
int dA = 5;
const int dSMax = 16;
int dS = 5;
const int dispAMax = 100;
int dispA = 10;
const int dispSMax = 100;
int dispS = 10;
const int pMax = 100;
int p = 90;
const int angleStepMax = 720;
int angleStep = 360;
const int numberOfLinesMax = 20;
int numberOfLines = 10;
const int maximumDistanceMax = 7;
int maximumDistance = 4;


//трекбар, измен€ющий первый порог
void onCannyTreshold1(int, void*)
{
	operatorCanny.changeCanny(cannyTreshhold1, cannyTreshhold2, operatorSobel_tb);
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}

//трекбар, измен€ющий второй порог
void onCannyTreshold2(int, void*)
{
	operatorCanny.changeCanny(cannyTreshhold1, cannyTreshhold2, operatorSobel_tb);
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}

//трекбар оператора собел€
void onOperatorSobel(int, void*)
{
	operatorCanny.changeCanny(cannyTreshhold1, cannyTreshhold2, operatorSobel_tb);
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}

//трекбар, мен€ющий шаг угла
void onAngleStep(int, void*) {}

//трекбар, мен€ющий dA
void ondA(int, void*) {}

//трекбар, мен€ющий dS
void ondS(int, void*) {}

//трекбар, мен€ющий dispA
void ondispA(int, void*) {}

//трекбар, мен€ющий dispS
void ondispS(int, void*) {}

//трекбар, мен€ющий dp
void onp(int, void*) {}

//трекбак, мен€ющий количества искомых границ на изображении
void onNumberOfLines(int, void*) {}

//трекбар, мен€ющий минимально допустимое рассто€ние между максимумами
void onMaximumDistance(int, void*) {}

//вывод информации о преобразовании радона на изображени€х
void myMouseCallbackinfo(int event, int x, int y, int flags, void* param)
{
	switch (event)
	{
		case 
			CV_EVENT_LBUTTONDOWN:
			{
				Mat showModel = transformRadon.showLinesOnModel(numberOfLines, maximumDistance);
				imshow(nameRadon.c_str(), showModel);

				Mat showFinalModelMat = cvarrToMat(operatorCanny.getImage()).clone();
				Mat showFinalModel = transformRadon.showFinalModel(showFinalModelMat, angleStep);
				imshow(nameOriginalImage.c_str(), showFinalModel);

			}
			break;
		case 
			CV_EVENT_RBUTTONDOWN:
			{
				Mat show = transformRadon.showLines(x, y, angleStep);
				imshow(cvNameCanny.c_str(), show);

			}
			break;
	}
}

//преобразование радона
void myMouseCallback(int event, int x, int y, int flags, void* param)
{
	switch (event)
	{
		case
			CV_EVENT_LBUTTONDOWN:
			{
				PHT transformRadonNew(operatorCanny, operatorSobel);
				Mat modelOriginal = transformRadonNew.transformRadon(dA, dS, 0.1*dispA, 0.1*dispS, 0.01*p, angleStep);
				writeMatToFile(modelOriginal, filenametxt);

				cvNamedWindow(nameRadon.c_str(), CV_WINDOW_AUTOSIZE);
				imshow(nameRadon, modelOriginal);

				transformRadon = transformRadonNew;

				destroyWindow(nameSearchLines);
				namedWindow(nameSearchLines, WINDOW_AUTOSIZE);
				resizeWindow(nameSearchLines, 600, 80);
				createTrackbar("Lines", nameSearchLines.c_str(), &numberOfLines, numberOfLinesMax, onNumberOfLines);
				createTrackbar("Distance", nameSearchLines.c_str(), &maximumDistance, maximumDistanceMax, onMaximumDistance);

				cvSetMouseCallback(nameRadon.c_str(), myMouseCallbackinfo);

				break;
			}
	}
}

//точка входа в класс Menu
void Menu::expectation()
{
	loadImage();
	trackBarCanny();
	trackBarRadon();
	cvSetMouseCallback(cvNameCanny.c_str(), myMouseCallback);
	waitKey(0);
}

//показ трекбаров фильтра Canny
void Menu::trackBarCanny()
{
	namedWindow(cvNameCannyTrackbar, WINDOW_AUTOSIZE);
	resizeWindow(cvNameCannyTrackbar, 600, 120);
	createTrackbar("Threshold1", cvNameCannyTrackbar.c_str(), &cannyTreshhold1, cannyTreshhold1Max, onCannyTreshold1);
	createTrackbar("Threshold2", cvNameCannyTrackbar.c_str(), &cannyTreshhold2, cannyTreshhold2Max, onCannyTreshold2);
	createTrackbar("Sobel op", cvNameCannyTrackbar.c_str(), &operatorSobel_tb, operatorSobelMax, onOperatorSobel);
}

//показ трекбаров преобразованного изображени€ Radon
void Menu::trackBarRadon()
{
	namedWindow(nameRadonTrackbar.c_str(), WINDOW_AUTOSIZE);
	resizeWindow(nameRadonTrackbar.c_str(), 800, 240);
	createTrackbar("dA", nameRadonTrackbar, &dA, dAMax, ondA);
	createTrackbar("dS", nameRadonTrackbar, &dS, dSMax, ondS);
	createTrackbar("dispA*0.1", nameRadonTrackbar, &dispA, dispAMax, ondispA);
	createTrackbar("dispS*0.1", nameRadonTrackbar, &dispS, dispSMax, ondispS);
	createTrackbar("p*0.01", nameRadonTrackbar, &p, pMax, onp);
	createTrackbar("Angle Step", nameRadonTrackbar, &angleStep, angleStepMax, onAngleStep);
}

//показ исходной картинки
void Menu::loadImage()
{
	cvNamedWindow(nameOriginalImage.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(nameOriginalImage.c_str(), operatorCanny.getImage());
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}





