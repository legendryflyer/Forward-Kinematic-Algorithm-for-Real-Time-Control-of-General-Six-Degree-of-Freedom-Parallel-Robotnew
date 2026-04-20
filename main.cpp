#include <iostream>
#include <vector>

#include "fk_model.h"

int main()
{
    try
    {

        // -------- manual prediction --------

        std::vector<double> legs =
            {450, 455, 448, 460, 452, 451};

        std::vector<double> pose =
            fk_predict(
                legs,
                "fk_predict.py");

        std::cout << "\nManual Prediction\n";

        std::cout << "X      : " << pose[0] << std::endl;
        std::cout << "Y      : " << pose[1] << std::endl;
        std::cout << "Z      : " << pose[2] << std::endl;
        std::cout << "Thetax : " << pose[3] << std::endl;
        std::cout << "Thetay : " << pose[4] << std::endl;
        std::cout << "Thetaz : " << pose[5] << std::endl;

        // -------- excel prediction --------

        fk_predict_excel(
            "fk_predict.py");

        std::cout << "\nExcel output saved to newfk_predictions.xlsx\n";
    }
    catch (const std::exception &e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}