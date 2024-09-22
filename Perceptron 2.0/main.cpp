#include <iostream>
#include <vector>
#include "perceptron.h"

int main() {
    size_t n_samples, n_features = 2;

    std::cout << "Enter number of samples: ";
    std::cin >> n_samples;

    std::vector<std::vector<double>> training_data(n_samples, std::vector<double>(n_features));
    std::vector<int> labels(n_samples);

    std::cout << "Enter " << n_samples << " data points with " << n_features << " features:\n";
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            std::cin >> training_data[i][j];
        }
    }

    std::cout << "Enter the labels (0 or 1) for each data point:\n";
    for (size_t i = 0; i < n_samples; ++i) {
        std::cin >> labels[i];
    }

    Perceptron perceptron(n_features, 0.01);
    int epochs = 1000;
    perceptron.train(training_data, labels, epochs);

    std::cout << "Model training complete. Enter inputs to predict:\n";
    std::vector<double> input(n_features);
    for (size_t i = 0; i < n_features; ++i) {
        std::cin >> input[i];
    }

    int prediction = perceptron.predict(input);
    std::cout << "Predicted output: " << prediction << std::endl;

    return 0;
}