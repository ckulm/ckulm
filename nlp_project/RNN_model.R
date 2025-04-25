
# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

# Load required libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(tidymodels)

# Load preprocessed data
load('data/claims-clean-example.RData')

# Partition data into training and testing sets
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels_binary <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
train_labels_multi <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels_binary <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
test_labels_multi <- testing(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

# Create and adapt the text vectorization layer
preprocess_layer <- layer_text_vectorization(
  max_tokens = 10000,
  output_sequence_length = 100
)
preprocess_layer %>% adapt(train_text)

# Preprocess data
train_sequences <- preprocess_layer(train_text)
test_sequences <- preprocess_layer(test_text)

# Define model for binary classification
binary_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 128, input_length = 100) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile binary model
binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train binary model
binary_model %>% fit(
  x = train_sequences,
  y = train_labels_binary,
  validation_split = 0.2,
  epochs = 11,
  batch_size = 32
)

# Save binary model
save_model_tf(binary_model, "results/rnn_binary_model")

# Define model for multi-class classification
multi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 128, input_length = 100) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = length(unique(train_labels_multi)), activation = 'softmax')

# Compile multi-class model
multi_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'sparse_categorical_accuracy'
)

# Train multi-class model
multi_model %>% fit(
  x = train_sequences,
  y = train_labels_multi,
  validation_split = 0.2,
  epochs = 11,
  batch_size = 32
)

# Save multi-class model
save_model_tf(multi_model, "results/rnn_multi_model")

# Generate predictions
binary_preds <- binary_model %>% predict(test_sequences) %>% round()

multi_preds <- multi_model %>% predict(test_sequences) %>% k_argmax()

# Format predictions into a data frame
class_labels <- levels(factor(claims_clean$mclass))

pred_df <- testing(partitions) %>%
  select(.id) %>%
  mutate(
    bclass.pred = ifelse(binary_preds == 1, "Positive", "Negative"),
    mclass.pred = class_labels[as.numeric(multi_preds) + 1]  # Correct indexing
  )

# Export predictions
write.csv(pred_df, "results/preds_df.csv", row.names = FALSE)

# Display results
print(head(pred_df))

### Predictions
### Using Claims-test

load('data/claims-test.RData')

# Load trained RNN models
binary_model <- load_model_tf("results/binary-model")
multi_model <- load_model_tf("results/multi-model")

# Convert text to numerical sequences using the trained preprocessing layer
test_sequences <- preprocess_layer(as.array(test_text))

# Generate binary predictions
binary_preds <- binary_model %>%
  predict(test_sequences) %>%
  round()  # Convert probabilities to 0/1 predictions

# Generate multi-class predictions
multi_preds <- multi_model %>%
  predict(test_sequences) %>%
  k_argmax()  # Convert probabilities to class indices

# Map predictions to class labels
class_labels_binary <- c("Negative", "Positive")  # Adjust based on your binary labels
class_labels_multi <- levels(factor(claims_clean$mclass))  # Multi-class labels from training

binary_pred_classes <- factor(binary_preds, labels = class_labels_binary)
multi_pred_classes <- factor(as.numeric(multi_preds) + 1, labels = class_labels_multi)

# Format predictions into a data frame
pred_df <- clean_df %>%
  select(.id) %>%
  mutate(
    bclass.pred = binary_pred_classes,
    mclass.pred = multi_pred_classes
  )

summary(pred_df)

# Save predictions
write.csv(pred_df, "results/rnn_test_predictions.csv", row.names = FALSE)
save(pred_df, file='results/preds_group14.RData')
# Display the predictions
print(head(pred_df))

### accuracy table 
library(dplyr)
library(yardstick)

# Combine test labels and predictions into a data frame
results <- testing(partitions) %>%
  select(.id) %>%
  mutate(
    true_bclass = test_labels_binary,  # True labels for binary classification
    true_mclass = test_labels_multi,  # True labels for multi-class classification
    bclass_pred = as.numeric(binary_preds),  # Binary predictions (0/1)
    mclass_pred = as.numeric(multi_preds)   # Multi-class predictions (index)
  )

# Convert binary labels to factors with appropriate levels
results <- results %>%
  mutate(
    true_bclass = factor(true_bclass, levels = 0:1, labels = c("Negative", "Positive")),
    bclass_pred = factor(bclass_pred, levels = 0:1, labels = c("Negative", "Positive")),
    true_mclass = factor(true_mclass, levels = 0:(length(class_labels) - 1), labels = class_labels),
    mclass_pred = factor(mclass_pred, levels = 0:(length(class_labels) - 1), labels = class_labels)
  )

three_metrics <- metric_set(accuracy, sensitivity, specificity)
binary_metrics <- three_metrics(results, truth = true_bclass, estimate = bclass_pred)
multiclass_metrics <- three_metrics(results, truth = true_mclass, estimate = mclass_pred)
