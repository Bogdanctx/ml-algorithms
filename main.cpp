#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <unordered_set>
#include <iostream>
#include "csv.h"

int rnd(int a, int b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(a, b);
    return distrib(gen);
}

struct Point {
    std::vector<std::string> columns;
    std::vector<double> attributes;

    std::string name;
    std::string artists;
};

std::ostream& operator<<(std::ostream& os, const Point& point) {
    os << point.name << ',' << point.artists;

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

    while (changed) {
        // Step 2 - assign points to clusters
        clusters = std::vector<std::vector<Point>>(numberOfClusters);

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

            clusters[centerIndex].push_back(x);
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
    }

    return {centers, clusters};
}

std::string sanitizeInput(std::string& input) {
    for (int i = 0; i < input.length(); i++) {
        if (input[i] == ',') {
            input[i] = ' ';
        }
    }
    return input;
}

int main() {
    csv::CSVReader reader("./tracks_features.csv");

    std::vector<Point> songs;

    std::cout << "Reading from file...\n";
    for (csv::CSVRow& row: reader) {
        Point song;

        song.columns = {"danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo", "time_signature"};
        for (const std::string& column: song.columns) {
            double value = row[column].get<double>();
            song.attributes.push_back(value);
        }


        song.artists = row["artists"].get<std::string>();
        song.name = row["name"].get<std::string>();

        song.artists = sanitizeInput(song.artists);
        song.name = sanitizeInput(song.name);

        songs.push_back(song);
    }
    std::cout << "Finished reading.\n";

    std::cout << "Applying clustering...\n";
    int numberOfClusters = 400;
    auto result = Lloyd(songs, numberOfClusters);
    std::ofstream fout("./clustering.csv");

    std::cout << "Clustering finished\n";

    fout << "cluster,name,artists";
    for (const std::string& column: songs[0].columns) {
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