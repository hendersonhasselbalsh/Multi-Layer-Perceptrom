#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <limits>
#include <cmath>
#include <numeric>
#include <string>


#ifndef MY_HEADER_H 
#define MY_HEADER_H


class Gnuplot {

	private:
		FILE* _gnuplotPipe;

	public:
		std::ofstream out;

		Gnuplot();
		~Gnuplot();

		void Plot(const std::string& str);
		void xRange(std::string xRangeStart, std::string xRangeEnd);
		void yRange(std::string yRangeStart, std::string yRangeEnd);
		void OutFile(const std::string path);
		void Grid(std::string x, std::string y);
		void CloseFile();


		friend Gnuplot& operator<<(Gnuplot& gnu, const std::string& str);
		friend Gnuplot& operator<<(Gnuplot& gnu, const int str);
		friend Gnuplot& operator<<(Gnuplot& gnu, const float str);
		friend Gnuplot& operator<<(Gnuplot& gnu, const double str);
	
};

#include "gnuplot-include.cpp"//

#endif //MY_HEADER_H
