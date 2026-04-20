#include "fk_model.h"

#include <cstdio>
#include <array>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::vector<double> fk_predict(
    const std::vector<double> &legs,
    const std::string &python_script)
{
    std::stringstream input;

    for (int i = 0; i < 6; i++)
    {
        input << legs[i];

        if (i < 5)
            input << " ";
    }

    std::string command =
        "echo \"" +
        input.str() +
        "\" | python " +
        python_script +
        " manual";

    std::array<char, 256> buffer;

    std::string result;

    FILE *pipe = popen(command.c_str(), "r");

    if (!pipe)
        throw std::runtime_error("python failed");

    while (fgets(buffer.data(), 256, pipe))
        result += buffer.data();

    pclose(pipe);

    std::vector<double> pose;

    std::stringstream ss(result);

    std::string value;

    while (getline(ss, value, ','))
        pose.push_back(std::stod(value));

    return pose;
}

void fk_predict_excel(
    const std::string &python_script)
{
    std::string command =
        "python " +
        python_script +
        " excel";

    std::array<char, 256> buffer;

    FILE *pipe = popen(command.c_str(), "r");

    if (!pipe)
        throw std::runtime_error("python failed");

    while (fgets(buffer.data(), 256, pipe))
        std::cout << buffer.data();

    pclose(pipe);
}