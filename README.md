# medicalDiagnosisUsingEHR
The project involves using Natural Language Processing Techniques and machine learning techniques to learn the word embeddings of the terms  in a patient's Electronic Health Record (EHR). 
SVM and Naive Bayes are then used to predict the medical speciality to which a patient's EHR belongs to, by utilizing the word embeddings learnt. 




The Dataset used - MTSamples.com(https://www.mtsamples.com/). A crawler and parser were developed to scrape data from the given website.  EHR related  to around 10 medical speciality were scraped from the website:
1. Cardiovascular / Pulmonary
2. ENT - Otolaryngology
3. Gastroenterology
4. Hematology - Oncology
5. Nephrology
6. Neurology
7. Obstetrics / Gynecology
8. Urology
9. Ophthalmology
10. Orthopedic


Initially, to get the baseline accuracy, we used the document as a whole, created a vector from the features extracted using Bag of Words and then SVM and Naive Bayes were used for making predictions.

Then Clamp (https://clamp.uth.edu/)  was utilized to extract medical terms from the data scraped.To extract data from the XML files created by Clamp, a XML Parser was built, which created the final csv files to be used for training and testing.


