#pragma once

#include "../utils/basic-includes.h"
#include "lost-function.h"



//-------------------------
// mean absolute error
//-------------------------
double MAE::f(double predicted, double correct)
{
	return std::abs(predicted - correct);
}

double MAE::df(double predicted, double correct)
{
	if (correct > predicted) { return 1.0; }
	else if (correct < predicted) { return -1.0; }
	else /*(correct == predicted)*/ { return 0.0; }
}




//-------------------------
// mean square error
//-------------------------
double MSE::f(double predicted, double correct)
{
	double error = predicted - correct;
	return std::pow(error, 2);
}

double MSE::df(double predicted, double correct)
{
	return 2 * (predicted - correct);
}




//-------------------------
// Root Mean Square error
//-------------------------
double RMSE::f(double predicted, double correct)
{
	return std::abs(predicted - correct);
}

double RMSE::df(double predicted, double correct)
{
	return (predicted - correct) / std::abs(predicted - correct);
}



//-------------------------
// cross entropy
//-------------------------
double CrossEntropy::f(double predicted, double correct)
{
	return correct * std::log(predicted);
}

double CrossEntropy::df(double predicted, double correct)
{
	double epsilon = 1e-10;
	// asure that predict is neither 0 or 1
	predicted = std::max(std::min(predicted, 1.0 - epsilon), epsilon);

	return - (correct / predicted) + ( (1 - correct) / (1 - predicted) );
}
