#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>

class Perceptron {
public:
    std::vector<double> weights;
    double bias;
    double learning_rate;

    Perceptron(size_t n_features, double learning_rate = 0.01)
        : weights(n_features, 0.0), bias(0.0), learning_rate(learning_rate) {}

    int activate(double x) {
        return x >= 0 ? 1 : 0;
    }

    // Predict the output for a given input
    int predict(const std::vector<double>& inputs) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * inputs[i];
        }
        return activate(sum);
    }

    // Update weights and bias based on the error
    void train(const std::vector<std::vector<double>>& training_data, const std::vector<int>& labels, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < training_data.size(); ++i) {
                int prediction = predict(training_data[i]);
                int error = labels[i] - prediction;
                for (size_t j = 0; j < weights.size(); ++j) {
                    weights[j] += learning_rate * error * training_data[i][j];
                }
                bias += learning_rate * error;
            }
        }
    }
};

#endif