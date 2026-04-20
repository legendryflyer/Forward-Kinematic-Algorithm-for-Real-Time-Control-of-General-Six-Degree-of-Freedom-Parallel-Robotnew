#pragma once

#include <vector>
#include <string>

// manual single prediction
std::vector<double> fk_predict(
    const std::vector<double> &legs,
    const std::string &python_script);

// excel batch prediction
void fk_predict_excel(
    const std::string &python_script);