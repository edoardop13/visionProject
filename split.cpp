#include <filesystem>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

void split_dataset(const fs::path& dataset_path, const fs::path& output_path, double train_ratio, double val_ratio) {
    // Create output directories
    fs::create_directories(output_path / "train");
    fs::create_directories(output_path / "val");
    fs::create_directories(output_path / "test");

    for (const auto& class_dir : fs::directory_iterator(dataset_path)) {
        if (class_dir.is_directory()) {
            std::vector<fs::path> images;
            for (const auto& file : fs::directory_iterator(class_dir)) {
                if (file.path().extension() == ".jpg") {
                    images.push_back(file.path());
                }
            }

            // Shuffle images
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(images.begin(), images.end(), g);

            // Calculate splits
            size_t train_cutoff = static_cast<size_t>(images.size() * train_ratio);
            size_t val_cutoff = static_cast<size_t>(images.size() * (train_ratio + val_ratio));

            // Distribute images
            auto copy_images = [&](const std::vector<fs::path>& imgs, const fs::path& dest_dir) {
                fs::create_directories(dest_dir);
                for (const auto& img : imgs) {
                    fs::copy(img, dest_dir / img.filename(), fs::copy_options::overwrite_existing);
                }
            };

            copy_images({images.begin(), images.begin() + train_cutoff}, output_path / "train" / class_dir.path().filename());
            copy_images({images.begin() + train_cutoff, images.begin() + val_cutoff}, output_path / "val" / class_dir.path().filename());
            copy_images({images.begin() + val_cutoff, images.end()}, output_path / "test" / class_dir.path().filename());
        }
    }
    std::cout << "Dataset split complete!" << std::endl;
}

int main() {
    fs::path dataset_path = "hagrid-sample/hagrid-sample-500k-384p/hagrid_500k"; // Replace with your dataset path
    fs::path output_path = "hagrid-sample/hagrid-sample-500k-384p/split";       // Replace with output path

    double train_ratio = 0.8;
    double val_ratio = 0.1;
    double test_ratio = 0.1;

    split_dataset(dataset_path, output_path, train_ratio, val_ratio);
    return 0;
}
