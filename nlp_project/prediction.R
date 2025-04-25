require(tidyverse)
require(keras)
require(tensorflow)
load('data/claims-test.RData')
load('data/claims-raw.RData')
source('scripts/preprocessing.R')
options(askYesNo = function(...) TRUE)

# apply preprocessing pipeline
clean_df <- claims_test %>%
  slice(1:100) %>%
  parse_data() %>%
  select(.id, text_clean)

# grab input
x <- clean_df %>%
  pull(text_clean)
x <- preprocess_layer(as.array(x))

# Generate binary predictions
binary_preds <- binary_model %>%
  predict(x) %>%
  round()  # Convert probabilities to 0/1 predictions

# Generate multi-class predictions
multi_preds <- multi_model %>%
  predict(x) %>%
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

save(pred_df, file = 'results/preds_group14.RData')