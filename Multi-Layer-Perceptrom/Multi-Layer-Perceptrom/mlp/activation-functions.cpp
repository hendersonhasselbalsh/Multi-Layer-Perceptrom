#pragma once

#include "activation-functions.h"


///-------------
///  custom
///-------------

CustonActivationFunction::CustonActivationFunction(std::function<double(double x)> f, double h)
    : _function(f), _h(h)
{
}

CustonActivationFunction::~CustonActivationFunction()
{
}

double CustonActivationFunction::f(double x)
{
    return _function(x);
}

double CustonActivationFunction::df(double x)
{
    // metodo das diferen�as finitas central de ordem h^2
    double df = (_function(x + _h)  -  _function(x - _h)) / (2.00 * _h);
    return df;
}

const char* CustonActivationFunction::ToString()
{
    return "custonMade";
}




///-------------
///  sigmoid
///-------------

Sigmoid::Sigmoid(double a)
{
    _a = a;
}

double Sigmoid::f(double x)
{
    return 1.0 / (1.0 + std::exp(-x*_a));
}

double Sigmoid::df(double x)
{
    double sig_x = f(x);
    return sig_x * (1.0 - sig_x);
}

const char* Sigmoid::ToString()
{
    return "Sigmoid";
}




///-------------
///  ReLU
///-------------

double ReLU::f(double x)
{
    return std::max(0.0, x);
}

double ReLU::df(double x)
{
    if (x < 0) { return 0.0; } else { return 1.0; };
}

const char* ReLU::ToString()
{
    return "ReLU";
}




///-------------
///  LeakyReLU
///-------------

double LeakyReLU::f(double x)
{
    return std::max(0.10 * x, x);
}

double LeakyReLU::df(double x)
{
    if (x < 0) { return 0.10; } else { return 1.0; };
}

const char* LeakyReLU::ToString()
{
    return "LeakyReLU";
}



///-------------
///  Tanh
///-------------

double Tanh::f(double x)
{
    return std::tanh(x);
}

double Tanh::df(double x)
{
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

const char* Tanh::ToString()
{
    return "Tanh";
}



///------------------
///  NormalizedTanh
///------------------

double NormalizedTanh::f(double x)
{
    return (std::tanh(x) + 1.0) / 2.0;
}

double NormalizedTanh::df(double x)
{
    double tanh_x = tanh(x);
    return (1.0 - tanh_x * tanh_x) / 2.0;
}

const char* NormalizedTanh::ToString()
{
    return "NormalizedTanh";
}



///------------------
///  NormalizedTanh
///------------------

double Linear::f(double x)
{
    return x;
}

double Linear::df(double x)
{
    return 1.0;
}

const char* Linear::ToString()
{
    return "Linear";
}



///------------------
///  NormalizedTanh
///------------------

double AdaptedSigmoid::f(double x)
{
    Sigmoid sigmoid;
    return (2.0 * sigmoid.f(x)) - 1.0;
}

double AdaptedSigmoid::df(double x)
{
    Sigmoid sigmoid;
    return 2.0 * std::exp(-x)  * sigmoid.f(x) * sigmoid.f(x);
}

const char* AdaptedSigmoid::ToString()
{
    return "AdaptedSigmoid";
}

