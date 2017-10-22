#include "stdafx.h"
#include<opencv2/opencv.hpp>

#include <iostream>
#include <ctype.h>
#include <fstream>
#include <string.h> 
#include <sstream> 
#include <vector>
#include <math.h>

#define PI 3.1415926525
#define _CRT_SECURE_NO_WARNINGS
#define fps 60
#pragma warning( disable : 4996 )

using namespace cv;
using namespace std;

int cnt = 0;
#define MAX_AREA 2500
Point pt[2000];
Point averageLocate[10];
double dist[9];
double angle[8];
int direction[8];
vector<int> num;

double Cosine(double x1, double y1, double x2, double y2);
int sixteenDirection(double x, double y);
void read_and_write_data_from_csv();

static void Readtxt() {
	int a = 0;
	stringstream ss;
	fstream fin;
	char buffert[100];
	char *token;
	string r[4];

	fin.open("test_01.txt", ios::in);

	while (fin.getline(buffert, sizeof(buffert),'\n')) {
		token = strtok(buffert, ",");
		
		int a = 0;
		while (token != NULL) {
			r[a] = token;
			token = strtok(NULL, ",");
			a++;
		}
		pt[cnt].x = atof(r[0].c_str());
		pt[cnt].y = atof(r[1].c_str());
		cnt++;
	}
	fin.close();

	return;
}

void analyze() 
{
	int cut = (cnt - 1) / 10;
	int flag = 0;
	int control_ruler = 0;
	for (int i = 0; i < 10; i++) {
		double x = 0 , y = 0;
		for (int j = 0; j < cut; j++) {
			x += pt[flag].x;
			y += pt[flag].y;
			flag++;
		}
		averageLocate[i].x = x/cut;
		averageLocate[i].y = y/cut;
	}

	for (int i = 0; i < 9; i++) {
		double x = averageLocate[i + 1].x - averageLocate[i].x;
		double y = averageLocate[i + 1].y - averageLocate[i].y;
		dist[i] = sqrt(x*x+y*y) * fps / cut;
	}
	for (int i = 0; i < 8; i++) {
		double x1 = averageLocate[i + 1].x - averageLocate[i].x;
		double y1 = averageLocate[i + 1].y - averageLocate[i].y;
		double x2 = averageLocate[i + 2].x - averageLocate[i + 1].x;
		double y2 = averageLocate[i + 2].y - averageLocate[i + 1].y;
		angle[i] = Cosine(x1, y1, x2, y2);
	}
	for (int i = 0; i < 9; i++) {
		direction[i] = sixteenDirection(averageLocate[i + 1].x - averageLocate[i].x, averageLocate[i + 1].y - averageLocate[i].y);
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

int sixteenDirection(double x,double y) 
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
			if (y/x <= tan(PI/16)) 
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

int main(int argc, const char** argv)
{
	Readtxt();

	analyze(); // 切出平均座標

	for (int i = 0; i < 10; i++) {
		cout << averageLocate[i].x << "," << averageLocate[i].y << endl;
	}
	cout << endl;
	for (int i = 0; i < 9; i++) {
		cout << dist[i] << endl;
	}
	cout << endl;
	for (int i = 0; i < 8; i++) {
		cout << angle[i] << endl;
	}
	cout << endl;
	for (int i = 0; i < 9; i++) {
		cout << direction[i] << endl;
	}

	read_and_write_data_from_csv();
	//system("pause");
	return 0;
}

//=========================================
//讀檔
void read_and_write_data_from_csv()
{
	vector<double> matrix;
	//readfile
	fstream file;
	file.open("training_data.csv",ios::in);
	string line;
	while (getline(file, line, '\n'))  //讀檔讀到跳行字元
	{
		istringstream templine(line); // string 轉換成 stream
		string data;
		while (getline(templine, data, ',')) //讀檔讀到逗號
		{
			matrix.push_back(atof(data.c_str()));  //string 轉換成數字
		}
	}
	file.close();

	cout << "How many turns?" << endl;
	int turn;
	cin >> turn;
	cout << "Abnormal or Normal? Abnormal = 1, Normal = 0" << endl;
	int inp;
	cin >> inp;

	if (inp == 1 || inp == 0) {
		//writefile
		file.open("training_data.csv", ios::out);
		int counter = 0;
		for (int i = 0; i<matrix.size(); i++)
		{
			file << matrix[i] << ",";
			counter++;
			if (counter == 28) {
				file << endl;
				counter = 0;
			}
		}
		for (int i = 0; i < 9; i++) {
			file << direction[i] << ",";
		}
		for (int i = 0; i < 9; i++) {
			file << dist[i] << ",";
		}
		for (int i = 0; i < 8; i++) {
			file << angle[i] << ",";
		}
		file << turn;
		file << inp;
		file.close();

	}

	
}