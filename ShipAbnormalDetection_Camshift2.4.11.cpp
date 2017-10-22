#include "stdafx.h"
#include <opencv2/opencv.hpp>  
#include <iostream>
#include <ctype.h>
#include <vector>

using namespace std;
using namespace cv;

#define PI 3.1415926525

const int N = 20; // 船有幾艘
Point averageLocate[N][10];
double dist[N][9];
double angle[N][8];
int direction[N][8];
int flagg[N] = {0};

double Cosine(double x1, double y1, double x2, double y2);
int sixteenDirection(double x, double y);
int cnt = 0;
void analyze(int cnt, int ship_ID);
int ReadCNN(int ship_ID);

//船隻資料
struct data {
	Rect selection;
	RotatedRect trackBox;
	Rect trackWindow;
	bool track = true;

	vector<Point2f>  pt;
	double L = 0;
	double R = 0;
	double angle_1 = 0;
	double angle_2 = 0;
	int suspicious_head;
	int suspicious_tail;
	bool get_suspicious = 0;
	int stage = 0;
	int wait_time = 0;

	vector<int>  close_time;
};

vector<struct data>input;

//追蹤部分參數
Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Mat backproj;
Mat mask;
int fps;
Mat frame;
int vmin = 20, vmax = 256, smin = 30;

#define MAX_AREA 2500
#define _CRT_SECURE_NO_WARNINGS
#pragma warning( disable : 4996 )


int main(int argc, const char** argv)
{
	VideoCapture cap;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;

	struct data i;

	//船隻資料寫入
/*
	test_case1
		[172, 156, 11, 11]

	test_cese2
		[82, 115, 12, 13]

	test_case3
		[393, 59, 11, 11]

	test_case4
		[172, 156, 11, 11]

	test_case5
		[271, 111, 11, 11]

	test_case6
		[192, 77, 11, 11]

	test_case7
		[201, 87, 11, 11]

	test_case9
		[272, 43, 17, 12]

*/
	i.selection = {192,77,11,11};
	input.push_back(i);

	trackObject = -1;

	//讀影片開始

	cap.open("D:\\testdata\\test_case6.avi");//影片抓取by路徑

	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		return -1;
	}
	namedWindow("CamShift Demo", 0);
	namedWindow("mask", WINDOW_NORMAL);
	
	//獲取fps 可能還有問題
	fps = cap.get(CV_CAP_PROP_FPS);
	if (fps > 500) {
		fps = 30;
	}
	//cout << fps << endl;

	Mat hsv, hue, hist, histimg = Mat::zeros(200, 320, CV_8UC3);
	bool paused = false;
	for (;;)
	{
		if (!paused)
		{
			cap >> frame;
			if (frame.empty())
				break;
		}
		//讀影片結束
		Mat gray, src;
		cvtColor(frame, src, cv::COLOR_RGB2GRAY);
		frame.copyTo(image);
		threshold(src, gray, 128, 255, cv::THRESH_BINARY);
		cvtColor(gray, image, cv::COLOR_GRAY2RGB);

		namedWindow("gray", WINDOW_NORMAL);
		imshow("gray", image);

		if (!paused)
		{
			cvtColor(image, hsv, COLOR_BGR2HSV);

			if (trackObject)//trackObject!=0
			{
				int _vmin = vmin, _vmax = vmax;

				inRange(hsv, Scalar(0, 0, 40),
					Scalar(180, 256, 256), mask);
				Mat black_img(image.rows, image.cols, CV_8UC1, Scalar(0));

				int ch[] = { 0, 0 };
				hue.create(hsv.size(), hsv.depth());
				mixChannels(&hsv, 1, &hue, 1, ch, 1);

				if (trackObject < 0)
				{
					for (int a = 0; a < input.size(); a++) {
						cv::rectangle(black_img, input[a].selection, cv::Scalar(255), -1);
						if (a == input.size() - 1)
							mask &= black_img;

						Mat roi(hue, input[a].selection), maskroi(mask, input[a].selection);
						calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
						normalize(hist, hist, 0, 255, NORM_MINMAX);
						input[a].trackWindow = input[a].selection;
					}
					trackObject = 1; // Don't set up again, unless user selects new ROI
				}
				//遮罩處理結束(包函追蹤目標物確定)

				for (int a = 0; a < input.size(); a++) { //抓取到非船隻的處理
					if (input[a].track == true) {
						cv::rectangle(black_img, input[a].trackWindow, cv::Scalar(255), -1);
						if (input[a].trackWindow.area() >= MAX_AREA)
						{
							cv::rectangle(black_img, input[a].trackWindow, cv::Scalar(0), -1);
							input[a].track = false;
						}
					}
				}
				mask &= black_img;
				imshow("mask", mask);

				// Perform CAMShift
				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
				backproj &= mask;


				//Camshift主要部分
				for (int a = 0; a < input.size(); a++) {

					if (input[a].track == true) {

						input[a].trackBox = CamShift(backproj, input[a].trackWindow,
							TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1)); //trackBox太小

						if (input[a].trackBox.center.x >= frame.cols || input[a].trackBox.center.x <= 0
							|| input[a].trackBox.center.y >= frame.rows || input[a].trackBox.center.y <= 0) {
							input[a].track == false;
						}
						else {
							if (input[a].selection.area() <= 1)
							{
								int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
								input[a].trackWindow = Rect(input[a].trackWindow.x - r, input[a].trackWindow.y - r,
									input[a].trackWindow.x + r, input[a].trackWindow.y + r) &
									Rect(0, 0, cols, rows);
							}

							if (backprojMode)
								cvtColor(backproj, frame, COLOR_GRAY2BGR);

							ellipse(frame, input[a].trackBox, Scalar(0, 255, 0), 1, 16);//畫橢圓

							if (input[a].pt.size() > 1200) {
								
								analyze(input[a].pt.size(), a);
								for (int i = 0; i < 9; i++)
									cout << direction[a][i] << " ";
								cout << endl;
								for (int i = 0; i < 9; i++)
									cout << dist[a][i] << " ";
								cout << endl;
								for (int i = 0; i < 8; i++)
									cout << angle[a][i] << " ";
								cout << endl;
								cout << ReadCNN(a) << endl;
								cout << endl;
								cout << endl;
								//cout << ReadCNN(a) << endl;
								
								if (ReadCNN(a) == 1) {
									flagg[a] ++;
								}
							}
							

							input[a].pt.push_back(input[a].trackBox.center);//畫出路徑部分
							if (input[a].pt.size() > 1) {
								for (int b = 0; b < input[a].pt.size() - 2; b++) {
									line(frame, input[a].pt[b], input[a].pt[b + 1], Scalar(0, 255, 0), 2);
								}
							}
						}
					}
				}
			}
		}

		else if (trackObject < 0)
			paused = false;

		imshow("CamShift Demo", frame);

		char c = (char)waitKey(5);

		if (c == 27)
			break;
		switch (c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 'c':
			trackObject = 0;
			histimg = Scalar::all(0);
			break;
		case 'h':
			showHist = !showHist;
			if (!showHist)
				destroyWindow("Histogram");
			else
				namedWindow("Histogram", 1);
			break;
		case 'p':
			paused = !paused;
			break;
		default:
			;
		}
	}

	return 0;
}

int ReadCNN(int ship_ID)
{
	const int CLASSES = 2;
	const int NUMBER_OF_TESTING_SAMPLES = 1;
	const int ATTRIBUTES_PER_SAMPLE = 26;
	Mat training_set_classifications(ATTRIBUTES_PER_SAMPLE, CLASSES, CV_32F, Scalar(0));
	Mat test_set_classifications(NUMBER_OF_TESTING_SAMPLES, CLASSES, CV_32F, Scalar(0));
	
	Mat classificationResult(1, CLASSES, CV_32F);
	

	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES_PER_SAMPLE;//input layer  
	layers.at<int>(1, 0) = 16;//hidden layer  

	layers.at<int>(2, 0) = CLASSES;//output layer  


	CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM, 1.0 / 10.0, 1);
	
	Mat test_set(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32F);
	for (int i = 0; i < 9; i++) {
		test_set.at<float>(0, i) = direction[ship_ID][i];
	}
	for (int i = 0; i < 9; i++) {
		test_set.at<float>(0, i+9) = dist[ship_ID][i];
	}
	for (int i = 0; i < 8; i++) {
		test_set.at<float>(0, i + 18) = angle[ship_ID][i];
	}
	//讀取 xml 檔，並用剛才輸入的欄位值，做分類，看看屬於哪個類別 
	cv::Mat test_sample;
	test_sample = test_set.row(0);
	CvFileStorage* storage2 = cvOpenFileStorage("nn_model.xml", 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage2, 0, "NN_Model");
	nnetwork.read(storage2, n);
	cvReleaseFileStorage(&storage2);
	nnetwork.predict(test_sample, classificationResult);

	float max = 0;
	int max_i = 0;
	for (int i = 0; i<CLASSES; i++) {
		//printf("第 %d 類神經元的值為：%.2f\n", i, classificationResult.at<float>(0, i));

		//以下三行取最大值
		if (max<classificationResult.at<float>(0, i)) {
			max = classificationResult.at<float>(0, i);
			max_i = i;
		}
	}
	return max_i;
}

void analyze(int cnt,int ship_ID) // cnt: 路徑記錄幾個點
{
	int cut = (cnt - 1) / 10;
	int flag = 0;
	int control_ruler = 0;
	for (int i = 0; i < 10; i++) {
		double x = 0, y = 0;

		for (int j = 0; j < cut; j++) {
			x += input[ship_ID].pt[flag].x;
			y += input[ship_ID].pt[flag].y;
			flag++;
		}
		averageLocate[ship_ID][i].x = x / cut;
		averageLocate[ship_ID][i].y = y / cut;
	}

	for (int i = 0; i < 9; i++) {
		double x = averageLocate[ship_ID][i + 1].x - averageLocate[ship_ID][i].x;
		double y = averageLocate[ship_ID][i + 1].y - averageLocate[ship_ID][i].y;
		dist[ship_ID][i] = sqrt(x*x + y*y)*fps / cut;
	}
	for (int i = 0; i < 8; i++) {
		double x1 = averageLocate[ship_ID][i + 1].x - averageLocate[ship_ID][i].x;
		double y1 = averageLocate[ship_ID][i + 1].y - averageLocate[ship_ID][i].y;
		double x2 = averageLocate[ship_ID][i + 2].x - averageLocate[ship_ID][i + 1].x;
		double y2 = averageLocate[ship_ID][i + 2].y - averageLocate[ship_ID][i + 1].y;
		angle[ship_ID][i] = Cosine(x1, y1, x2, y2);
	}
	for (int i = 0; i < 9; i++) {
		direction[ship_ID][i] = sixteenDirection(averageLocate[ship_ID][i + 1].x - averageLocate[ship_ID][i].x, averageLocate[ship_ID][i + 1].y - averageLocate[ship_ID][i].y);
	}
}

double Cosine(double x1, double y1, double x2, double y2)
{
	double cross = x1*x2 + y1*y2;
	double length_a = sqrt(x1*x1 + y1*y1);
	double length_b = sqrt(x2*x2 + y2*y2);
	if (length_a == 0 || length_b == 0)
		return 1;
	else
		return cross / (length_a*length_b);
}

int sixteenDirection(double x, double y)
{
	if (x == 0 && y == 0) return 0;
	else if (x == 0 || y == 0) {
		if (x == 0) {
			if (y > 0) return 13;
			else if (y < 0) return 15;
		}
		else {
			if (x > 0) return 1;
			else if (x < 0) return 9;
		}
	}
	else
	{
		if (x > 0 && y > 0) // 第一象限
		{
			if (y / x <= tan(PI / 16))
				return 1;
			else if (y / x <= tan(3 * PI / 16))
				return 16;
			else if (y / x <= tan(5 * PI / 16))
				return 15;
			else if (y / x <= tan(7 * PI / 16))
				return 14;
			else
				return 13;
		}
		else if (x < 0 && y > 0)  //第二象限
		{
			y *= -1;
			if (y / x <= tan(PI / 16))
				return 9;
			else if (y / x <= tan(3 * PI / 16))
				return 10;
			else if (y / x <= tan(5 * PI / 16))
				return 11;
			else if (y / x <= tan(7 * PI / 16))
				return 12;
			else
				return 13;
		}
		else if (x < 0 && y < 0)
		{
			if (y / x <= tan(PI / 16))
				return 9;
			else if (y / x <= tan(3 * PI / 16))
				return 8;
			else if (y / x <= tan(5 * PI / 16))
				return 7;
			else if (y / x <= tan(7 * PI / 16))
				return 6;
			else
				return 5;
		}
		else if (x > 0 && y < 0)
		{
			y *= -1;
			if (y / x <= tan(PI / 16))
				return 1;
			else if (y / x <= tan(3 * PI / 16))
				return 2;
			else if (y / x <= tan(5 * PI / 16))
				return 3;
			else if (y / x <= tan(7 * PI / 16))
				return 4;
			else
				return 5;
		}
	}
}