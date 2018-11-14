/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"
#include "catdog_parser.h"

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
  // clang-format off
static const bool tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
// clang-format on
#undef O
#undef X

  // construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  // clang-format off
  using fc = tiny_dnn::layers::fc;
  using conv = tiny_dnn::layers::conv;
  using ave_pool = tiny_dnn::layers::ave_pool;
  using tanh = tiny_dnn::activation::tanh;

  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;

  nn << conv(64, 64, 5, 1, 6,   // C1, 1@64x64-in, 6@60x60-out
             padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << ave_pool(60, 60, 6, 2)   // S2, 6@60x60-in, 6@30x30-out
     << tanh()
     << conv(30, 30, 5, 6, 16,   // C3, 6@30x30-in, 16@26x26-out
             connection_table(tbl, 6, 16),
             padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << ave_pool(26, 26, 16, 2)  // S4, 16@26x26-in, 16@13x13-out
     << tanh()
     << conv(13, 13, 6, 16, 120,   // C5, 16@13x13-in, 120@8x8-out
             padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << ave_pool(8, 8, 120, 2)  //S6, 120@8x8-in, 120@4x4-out
     << tanh()
     << conv(4, 4, 4, 120, 20,  //C7, 120@4x4-in, 20@1x1-out
             padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << fc(20, 2, true, backend_type)  // F8, 20-in, 2-out
     << tanh();
}

static void train_lenet(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch,
                        tiny_dnn::core::backend_t backend_type) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adagrad optimizer;

  construct_net(nn, backend_type);

  std::cout << "load models..." << std::endl;

  // load MNIST dataset
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;

  //tiny_dnn::parse_train_images("./data/train/", train_images, train_labels, 8, 32, 32, 2, 2, -1, 1, 0);
  tiny_dnn::parse_train_images("./data/train/", train_images, train_labels, 8000, 64, 64, 2, 2, -1, 1, 0);
  tiny_dnn::parse_train_images("./data/train/", test_images, test_labels, 2000, 64, 64, 2, 2, -1, 1, 8000);

//  for(int j = 0; j < 32; j++) {
//      for(int i = 0; i < 32; i++) {
//
//          int index = i + j * 32;
//          std::cout<<std::setw(4)<<std::setprecision(1)<<test_images[0][index]<<" ";
//      }
//      std::cout<<std::endl;
//  }
//
//  std::cout<<"\n";
//
//  std::cout<<test_labels[0]<<std::endl;


  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
  // save network model & trained weights
  nn.save("LeNet-model");
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {{
    "internal", "nnpack", "libdnn", "avx", "opencl",
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epochs 30"
            << " --minibatch_size 16"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 1;
  int epochs                             = 30;
  std::string data_path                  = "";
  int minibatch_size                     = 16;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (data_path == "") {
    std::cerr << "Data path not specified." << std::endl;
    usage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 60000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (60000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  return 0;
}
