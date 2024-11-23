#pragma once
#include "gnuplot-include.h"



Gnuplot::Gnuplot()
{
	_gnuplotPipe  =  _popen("gnuplot -persist", "w");
}



Gnuplot::~Gnuplot()
{
	_pclose(_gnuplotPipe);

	if (out.is_open()) { out.close(); }
}



void Gnuplot::Plot(const std::string& str)
{
	std::string comand = "plot ";
	comand += str;

	fprintf(_gnuplotPipe, comand.c_str());
}

void Gnuplot::xRange(std::string xRangeStart, std::string xRangeEnd)
{
	std::string comand = "set xrange [" + xRangeStart + ":" + xRangeEnd +"] \n";
	fprintf(_gnuplotPipe, comand.c_str());
}


void Gnuplot::yRange(std::string yRangeStart, std::string yRangeEnd)
{
	std::string comand = "set yrange [" + yRangeStart + ":" + yRangeEnd +"] \n";
	fprintf(_gnuplotPipe, comand.c_str());
}


void Gnuplot::Grid(std::string x, std::string y)
{
	std::string comand = "set grid\nset ytics " + y + "\n" + "set xtics " + x + "\n";
	fprintf(_gnuplotPipe, comand.c_str());
}


void Gnuplot::OutFile(const std::string path)
{
	out = std::ofstream(path.c_str());
}



void Gnuplot::CloseFile()
{
	out.close();
}




Gnuplot& operator<<(Gnuplot& gnu, const std::string& str)
{
	fprintf(gnu._gnuplotPipe, str.c_str());
	return gnu;
}

Gnuplot& operator<<(Gnuplot& gnu, const int str)
{
	std::string value  =  std::to_string(str);
	fprintf(gnu._gnuplotPipe, value.c_str());
	return gnu;
}

Gnuplot& operator<<(Gnuplot& gnu, const float str)
{
	std::string value  =  std::to_string(str);
	fprintf(gnu._gnuplotPipe, value.c_str());
	return gnu;
}

Gnuplot& operator<<(Gnuplot& gnu, const double str)
{
	std::string value  =  std::to_string(str);
	fprintf(gnu._gnuplotPipe, value.c_str());
	return gnu;
}
