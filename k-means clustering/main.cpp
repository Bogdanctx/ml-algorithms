#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <unordered_set>
#include <iostream>
#include <omp.h>
#include "../csv.h"

int rnd(int a, int b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(a, b);
    return distrib(gen);
}

struct Point {
    std::vector<std::string> columns;
    std::vector<std::string> extraAttributes;
    std::vector<double> attributes;
};

std::ostream& operator<<(std::ostream& os, const Point& point) {
    const int length = point.extraAttributes.size();

    for(int i = 0; i < length; i++) {
        os << point.extraAttributes[i];
        if(i + 1 < length) {
            os << ',';
        }
    }

    for (const double& attr : point.attributes) {
        os << "," << attr;
    }

    return os;
}

double square(double x) {
    return x * x;
}

double euclidianDistance(Point x, Point y) {
    double sum = 0;
    for (int d = 0; d < x.attributes.size(); d++) {
        sum += square(x.attributes[d] - y.attributes[d]);
    }
    return sum;
}

std::pair<std::vector<Point>, std::vector<std::vector<Point>>> Lloyd(const std::vector<Point>& points, int numberOfClusters) {
    // Step 1 -> Initialize centers
    std::vector<Point> centers;
    std::unordered_set<int> usedCenters;
    while (centers.size() < numberOfClusters) {
        int randomIndex = rnd(0, points.size() - 1);

        // center already used
        if (usedCenters.find(randomIndex) != usedCenters.end()) {
            continue;
        }

        centers.push_back(points[randomIndex]);
        usedCenters.insert(randomIndex);
    }

    bool changed = true;
    std::vector<std::vector<Point>> clusters;
    std::vector<Point> oldCenters;

    int iteration = 0;
    while (changed) {
        // Step 2 - assign points to clusters
        clusters = std::vector<std::vector<Point>>(numberOfClusters);
        std::cout << "Begin iteration " << iteration << std::endl;
        
        #pragma omp parallel for
        for (const Point& x: points) {
            double minDistance = euclidianDistance(x, centers[0]);
            int centerIndex = 0;
            for (int i = 1; i < centers.size(); i++) {
                double distance = euclidianDistance(x, centers[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    centerIndex = i;
                }
            }

            #pragma omp critical
            {
                clusters[centerIndex].push_back(x);
            }
        }

        // Step 3 - Stop if no change
        if (!oldCenters.empty()) {
            changed = false;
            for (int i = 0; i < numberOfClusters; i++) {
                if (euclidianDistance(centers[i], oldCenters[i]) > 1e-6) {
                    changed = true;
                }
            }
        }

        // Step 4 - recompute centers as centroid
        oldCenters = centers;
        for (int l = 0; l < numberOfClusters; l++) {
            std::vector<Point> cluster = clusters[l];

            Point newCenter;
            newCenter.attributes.resize(points[0].attributes.size(), 0);
            for (const Point& p: cluster) {
                for (int d = 0; d < p.attributes.size(); d++) {
                    newCenter.attributes[d] += p.attributes[d];
                }
            }
            for (int d = 0; d < newCenter.attributes.size(); d++) {
                newCenter.attributes[d] /= cluster.size();
            }

            centers[l] = newCenter;
        }

        std::cout << "End iteration " << iteration++ << std::endl;
    }

    return {centers, clusters};
}

std::string sanitizeInput(std::string input) {
    for (int i = 0; i < input.length(); i++) {
        if (input[i] == ',') {
            input[i] = ' ';
        }
    }
    return input;
}

int main() {
    csv::CSVReader reader("./data/spotify.csv");

    std::vector<Point> points;
    const std::vector<std::string> extraAttributes = {"track_name", "track_artist"};

    std::cout << "Reading from file...\n";
    for (csv::CSVRow& row: reader) {
        Point point;

        point.columns = {"danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo"};
        for (const std::string& column: point.columns) {
            double value = row[column].get<double>();
            point.attributes.push_back(value);
        }

        for(const std::string& extra: extraAttributes) {
            const std::string sanitized = sanitizeInput(row[extra].get<std::string>());

            point.extraAttributes.push_back(sanitized);
        }

        points.push_back(point);
    }
    std::cout << "Finished reading.\n";

    std::cout << "Applying clustering...\n";
 
    int numberOfClusters = 400;
    auto result = Lloyd(points, numberOfClusters);
 
    std::ofstream fout("./clustering.csv");

    std::cout << "Clustering finished\n";

    fout << "cluster";
    for(const std::string& extra: points[0].extraAttributes) {
        fout << ',' << extra;
    }
    for (const std::string& column: points[0].columns) {
        fout << ',' << column;
    }
    fout << std::endl;

    for (int c = 0; c < result.first.size(); c++) {
        const std::vector<Point> cluster = result.second[c];

        for (Point p: cluster) {
            fout << c + 1 << ',' << p << std::endl;
        }
    }

    return 0;
}