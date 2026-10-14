#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <vector>

/**
 * Parameters for N-body equations of motion, compatible for GSL
 */
struct ParamsNbody {
    double et0;
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double A;
    double m;
    double rhoS;
    double rhoD;
    double LU;
    double TU;
};

#endif // PARAMS_H