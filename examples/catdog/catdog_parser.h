/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/util.h"

#include "opencv2/opencv.hpp"

namespace tiny_dnn {

    using namespace cv;

    inline void get_pixel(Mat &image, vec_t &data, int width, int height,
                           int x_padding,
                           int y_padding,
                           float_t scale_min,
                           float_t scale_max) 
    {
        if(2 * x_padding >= width || 2 * y_padding >= height ||
           x_padding < 0 || y_padding < 0) {
            throw nn_error("Image file padding too large");
        } 

        if(scale_min >= scale_max)
            throw nn_error("scale_max must be greater than scale_min");

        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        Mat gray_resize;
        resize(gray, gray_resize, Size(width-2*x_padding,height-2*y_padding), 0, 0, INTER_LINEAR);
//        imshow("atom_window", gray_resize);
//        waitKey(0);

        data.resize(width * height, scale_min);
        for(int r = 0; r < gray_resize.rows; ++r) {
            for(int c = 0; c < gray_resize.cols; ++c) {

                data[c + x_padding + width * ( r + y_padding )] = 
                    gray_resize.at<unsigned char>(r, c)/float_t(255) * 
                    (scale_max - scale_min) + scale_min;
            }
        }
    }

    void parse_test_images(const std::string &image_path,
                           std::vector<vec_t> &test_image,
                           int num,
                           int width,
                           int height,
                           int x_padding,
                           int y_padding,
                           float_t scale_min,
                           float_t scale_max)
    {
        test_image.resize(num);

        for(int i = 0; i < num; i++) {
            std::string test_name = image_path + std::to_string(i) + ".jpg";

            Mat image;
            image = imread(test_name, IMREAD_COLOR);
            if(!image.data)
            {
                std::cout<<"No image data."<<std::endl;
                exit(0);
            }

            vec_t data;
            get_pixel(image, data, width, height, x_padding, y_padding, scale_min, scale_max);

            test_image[i] = data;

        }
    }

    void parse_train_images(const std::string &image_path,
                            std::vector<vec_t> &train_image,
                            std::vector<label_t> &train_label,
                            int num,
                            int width,
                            int height,
                            int x_padding,
                            int y_padding,
                            float_t scale_min,
                            float_t scale_max, 
                            int start_num)
    {
        train_image.resize(num * 2);
        train_label.resize(num * 2);

        for(int i = start_num; i < start_num + num; i++) {
            std::string cat_name = image_path + "cat." + std::to_string(i) + ".jpg";
            std::string dog_name = image_path + "dog." + std::to_string(i) + ".jpg";

            Mat image_cat, image_dog;
            image_cat = imread(cat_name, IMREAD_COLOR);
            image_dog = imread(dog_name, IMREAD_COLOR);
            if(!image_cat.data || !image_dog.data) {
                std::cout<<"Image reading error."<<std::endl;
                std::cout<<"Image Id: "<< i <<std::endl;
                exit(0);
            }

            vec_t data_cat;
            vec_t data_dog;
            get_pixel(image_cat, data_cat, width, height, x_padding, y_padding, scale_min, scale_max);
            get_pixel(image_dog, data_dog, width, height, x_padding, y_padding, scale_min, scale_max);
            train_image[2 * (i - start_num)] = data_cat;
            train_image[2 * (i - start_num) + 1] = data_dog;
            label_t label_cat = 0; // 1 mean cat ; 0 means dog
            label_t label_dog = 1; 

            train_label[2 * (i - start_num)] = label_cat;
            train_label[2 * (i - start_num) + 1] = label_dog;
        }
        
    }

}  // namespace tiny_dnn
