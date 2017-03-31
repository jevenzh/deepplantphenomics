import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, save_checkpoints=False, report_rate=40)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(2)
model.set_number_of_threads(8)
model.set_image_dimensions(179, 998, channels)
model.set_resize_images(True)

model.set_problem_type('regression')
model.set_num_regression_outputs(1)
model.set_train_test_split(0.8)
model.set_learning_rate(0.0001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(200)

# Augmentation options
model.set_augmentation_brightness_and_contrast(True)
model.set_augmentation_flip_horizontal(True)
model.set_augmentation_flip_vertical(True)
model.set_augmentation_crop(True, crop_ratio=0.9)

# Load all VIS images from a Lemnatec image repository
model.load_multiple_labels_from_csv('./data/germ-wide/leaf_counts.csv')
model.load_images_with_ids_from_directory('./data/germ-wide')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[11, 11, 3, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_fully_connected_layer(output_size=1024, activation_function='relu')

model.add_output_layer()

# Begin training the regression model
model.begin_training()
