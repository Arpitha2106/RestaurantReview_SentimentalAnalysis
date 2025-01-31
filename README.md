# **Sentiment Analysis on Restaurant Reviews**

## **Overview**
This project performs **sentiment analysis** on restaurant reviews using **two different text preprocessing techniques**:
- **WordNet Lemmatization**
- **Porter Stemming**

The aim is to compare their impact on sentiment classification performance across multiple machine learning models.
## **Project Structure**
- `Review_WordNetLemmatizer.ipynb` → Uses **Lemmatization** for text preprocessing.
- `Review_with_PorterStemmer.ipynb` → Uses **Stemming** for text preprocessing.
- `Restaurant_Reviews.tsv` → Dataset containing **1,000 restaurant reviews** labeled as positive (1) or negative (0).

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:** NLTK, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Models:** Logistic Regression, Random Forest, SVM
- **Vectorization Techniques:** CountVectorizer, TF-IDF

## **Work Flow**
1. **Load Dataset:** Read the restaurant reviews dataset.
2. **Preprocessing:**
   - **Lemmatization Approach:** Converts words to their dictionary form.
   - **Stemming Approach:** Reduces words to their root form.
3. **Feature Extraction:** Convert text into numerical format using **CountVectorizer** and **TF-IDF**.
4. **Train Machine Learning Models:** Use **Logistic Regression, Random Forest, and SVM**.
5. **Evaluate Performance:** Compare models based on **accuracy and confusion matrix**.

## **Results Summary**
| Model                 | Lemmatization (Accuracy) | Stemming (Accuracy) |
|-----------------------|------------------------|--------------------|
| Logistic Regression (Count) | **73%** | 71% |
| Logistic Regression (TF-IDF) | 74% | **75.5%** |
| Random Forest (Count) | 68.5% | **70%** |
| Random Forest (TF-IDF) | 70% | **73%** |
| SVM (Count) | 71% | **72%** |
| SVM (TF-IDF) | **74%** | 73% |

