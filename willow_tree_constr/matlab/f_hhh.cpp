#include <iostream>
#include <random>
#include <cmath>
#include <vector>

// Function to calculate the skewness of a vector
double skewness(const std::vector<double>& data) {
    double n = data.size();
    double mean = 0.0, M2 = 0.0, M3 = 0.0;

    for (double x : data) {
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        M2 += delta * delta2;
        M3 += delta2 * delta2 * delta;
    }

    double skew = (sqrt(n) * M3) / pow(M2, 1.5);
    return skew;
}

// Function to calculate the kurtosis of a vector
double kurtosis(const std::vector<double>& data) {
    double n = data.size();
    double mean = 0.0, M2 = 0.0, M4 = 0.0;

    for (double x : data) {
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        M2 += delta * delta2;
        M4 += delta2 * delta2 * delta2 * delta;
    }

    double kurt = (n * M4) / (M2 * M2) - 3.0;
    return kurt;
}

// Function to generate a random normal distribution
std::vector<double> normal_distribution(double mean, double stddev, int size) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{mean, stddev};
    std::vector<double> data(size);

    for (int i = 0; i < size; ++i) {
        data[i] = d(gen);
    }

    return data;
}

// Function to calculate the inverse of the standard normal CDF
double normal_ppf(double p) {
    double a1 = -3.969683028665376e+01;
    double a2 =  2.209460984245205e+02;
    double a3 = -2.759285104469687e+02;
    double a4 =  1.383577518672690e+02;
    double a5 = -3.066479806614716e+01;
    double a6 =  2.506628277459239;

    double b1 = -5.447609879822406e+01;
    double b2 =  1.615858368580409e+02;
    double b3 = -1.556989798598866e+02;
    double b4 =  6.680131188771972e+01;
    double b5 = -1.328068155288572e+01;

    double c1 = -7.784894002430293e-03;
    double c2 = -3.223964580411365e-01;
    double c3 = -2.400758277161838;
    double c4 = -2.549732539343734;
    double c5 =  4.374664141464968;
    double c6 =  2.938163982698783;

    double d1 =  7.784695709041462e-03;
    double d2 =  3.224671290700398e-01;
    double d3 =  2.445134137142996;
    double d4 =  3.754408661907416;

    double p_low = 0.02425;
    double p_high = 1 - p_low;

    double q, r;

    if (0 < p && p < p_low) {
        q = sqrt(-2 * log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    } else if (p_low <= p && p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
    } else if (p_high < p && p < 1) {
        q = sqrt(-2 * log(1 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    } else {
        return 0;
    }
}

// Function to replicate the functionality of your provided code
void f_hhh(double mu, double sd, double ka3, double ka4, double& a, double& b, double& d, double& c, std::vector<double>& X) {
    int size = 1000000;
    std::vector<double> Z = normal_distribution(0, 1, size);
    X.resize(size);

    double delta = sd * sd;
    std::vector<double> data = normal_distribution(mu, sd, size);
    double gamma = skewness(data);
    double epsilon = kurtosis(data) - 3;
    a = gamma / sqrt(epsilon * delta);
    b = delta / sqrt(epsilon);
    c = mu - a * b;
    d = sqrt(epsilon) * sd;

    for (int i = 0; i < size; ++i) {
        X[i] = c + d * normal_ppf((Z[i] - a) / b);
    }
}
