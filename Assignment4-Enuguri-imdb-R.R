library(keras)
library(tidyverse)
set.seed(123)
n_sample <- 5000; maxlen <- 150; max_features <- 5000
imdb = dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb # Loads the data
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
train_sample = sample(1:nrow(x_train), n_sample)
test_sample = sample(1:nrow(x_test), n_sample)
x_train <- x_train[train_sample,] # focus on a subset of train sample
y_train <- y_train[train_sample] # focus on a subset of train sample
x_test <- x_test[test_sample,] # focus on a subset of test sample
y_test <- y_test[test_sample] # focus on a subset of test sample


write_rds(x_train, "X_train.rds")
write_rds(y_train, "Y_train.rds")
write_rds(x_test, "X_test.rds")
write_rds(y_test, "Y_test.rds")


#### Reading the Train & Test IMDB sets.
x_train <- read_rds("X_train.rds")
y_train <- read_rds("Y_train.rds")
x_test <- read_rds("X_test.rds")
y_test <- read_rds("Y_test.rds")

# Fit a simple RNN model
model_simple_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model_simple_rnn)



model_simple_rnn %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

history_rnn <- model_simple_rnn %>% fit(
  x_train, y_train,
  epochs = 8,
  batch_size = 120,
  validation_split = 0.3
)
#model_simple_rnn %>% evaluate(x_test,y_test)

model_simple_rnn %>% save_model_hdf5("model_simple_rnn.h5")
write_rds(history_rnn, "model_simple_rnn_history.rds")


##LSTM

# Define the model
model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>%
  layer_lstm(units = 64) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_lstm %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)
# Train the model
history_lstm <- model_lstm  %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 10,
  validation_split = 0.2
)

# Evaluate the model on the test set
#results <-model_lstm  %>% evaluate(x_test, y_test)


model_lstm %>% save_model_hdf5("model_lstm.h5")
write_rds(history_lstm, "model_lstm_history.rds")


##GRU


# Define the model
model_gru <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>%
  layer_gru(units = 64) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_gru %>% compile(
  loss = "binary_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)
# Train the model
history_gru <- model_gru %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 8,
  validation_split = 0.25
)

# Evaluate the model on the test set
#results <- model_gru %>% evaluate(x_test, y_test)



model_gru %>% save_model_hdf5("model_gru.h5")
write_rds(history_gru, "model_gru_history.rds")


#Bidirectional LSTM

# Define the model
model_bi_dir_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>%
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_bi_dir_lstm %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Train the model
history_bi_dir_lstm  <- model_bi_dir_lstm %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 10,
  validation_split = 0.2
)

# Evaluate the model on the test set
#results4 <- model_bi_dir_lstm %>% evaluate(x_test, y_test)



model_bi_dir_lstm %>% save_model_hdf5("model_bi_dir_lstm.h5")
write_rds(history_bi_dir_lstm , "model_bi_dir_lstm.rds")

#Bidirectinal GRU

# Define the model
model_bi_dir_gru<- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>%
  bidirectional(layer_gru(units = 64)) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_bi_dir_gru %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Train the model
model_bi_dir_gru_history <- model_bi_dir_gru  %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 10,
  validation_split = 0.2
)

# Evaluate the model on the test set
#results5 <- model_bi_dir_gru  %>% evaluate(x_test, y_test)


model_bi_dir_gru %>% save_model_hdf5("model_bi_dir_gru.h5")
write_rds(model_bi_dir_gru_history, "model_bi_dir_gru_history.rds")

#1 D conv

# Define the model
model_1d_convent <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_1d_convent %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Train the model
history_1d <- model_1d_convent %>% fit(
  x_train, y_train,
  batch_size = 256,
  epochs = 6,
  validation_split = 0.2
)

#results6 <- model_1d_convent %>% evaluate(x_test, y_test)


model_1d_convent %>% save_model_hdf5("model_1d_convent.h5")
write_rds(history_1d, "model_1d_history.rds")