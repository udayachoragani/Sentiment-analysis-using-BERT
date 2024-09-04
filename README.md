# Sentiment-analysis-using-BERT
Sentiment Analysis using BERT
Topic : LLM Fine-Tuning comparison study.
 
Selection Criteria for the base LLM

Contextual Understanding : Sentiment analysis often depends on understanding subtle nuances in text, like the impact of negation ("not good" vs. "good") or the overall sentiment in complex sentences. BERT's bidirectional approach helps it capture these nuances more effectively than models that only read text in one direction.

Pretraining on Diverse Data : BERT was pretrained on a massive and diverse corpus, including the English Wikipedia and Books-Corpus. This broad pretraining helps BERT develop a deep understanding of language.

Handling of Long Sentences : BERT uses the Transformer architecture, which includes an attention mechanism that allows the model to focus on important words or phrases in a sentence, enhancing its ability to detect sentiment even in the presence of distracting or irrelevant information.


 
Task specific considerations for fine tuning

Data Balance : Ensure training and validation splits maintain a 50/50 balance.

Text Cleaning : Remove HTML tags, special characters, and excessive whitespace.

Sequence Length : Choose an appropriate maximum sequence length, considering resource constraints.

Tokenization : Use BERT's Word-Piece tokenizer for converting text into tokens.

Padding : Apply padding and truncation to ensure uniform sequence lengths.

Number of Epochs: Fine-tune for 3 to 5 epochs, monitoring validation performance.


 
Data Preparation and Preprocessing Steps

Loading the Dataset: Download and load the dataset. Here we are using IMBD movie review dataset, which typically consists of 50,000 movie reviews split equally into positive and negative sentiment labels.

Data preprocessing : Remove HTML Tags, Convert all text to lowercase, Remove
Special Characters and excessive whitespace.

Tokenization : Use BERT's tokenizer(bert-base-uncased) to convert each review into tokens that BERT can process. The tokenizer splits words into subwords and handles special tokens.

Convert Tokens to IDs : Convert the tokenized words into their corresponding token IDs using BERT’s vocabulary. This step transforms the tokens into numerical input that BERT can process.
 
Data preparation and preprocessing steps

Data Splitting : Split the dataset into training and validation sets, typically using an 80/20 split. Ensure that both sets maintain the balance of positive and negative reviews.

Label Encoding : Convert sentiment labels(e.g., "positive" and "negative“) into numerical format (1 for positive and 0 for negative) which the model can process.

Data Batching : Organize the tokenized and encoded data into batches. Use a batch size that your hardware can handle (e.g., 16 or 32). Batching helps in efficient training by processing multiple reviews simultaneously.






 
Fine tuning and hyper parameter optimization strategies



Learning Rate : Use a learning rate warm-up strategy, where the learning rate gradually increases
for a few steps before stabilizing, to help the model adjust and avoid large initial weight updates.

Batch Size : Use a batch size between 16 and 32, depending on your available GPU memory. Larger batch sizes can improve training stability but require more memory.

Number of Epochs : Fine-tune for 3 to 5 epochs. Fewer epochs are usually sufficient when starting with a pre-trained BERT model. Monitor performance metrics on the validation set to decide when to stop training.

Optimizer : Use the AdamW optimizer, which is commonly used with BERT. AdamW incorporates weight decay, which helps in regularizing the model by preventing the weights from growing too large.

 
Evaluation metrics and performance analysis


Before Fine-tuning

•	Accuracy: 85%

•	Precision: 0.86

•	Recall: 0.84

•	F1-Score: 0.85

After Fine-tuning
 
•	Accuracy: 93%

•	Precision: 0.92

•	Recall: 0.93

•	F1-Score: 0.93
 
