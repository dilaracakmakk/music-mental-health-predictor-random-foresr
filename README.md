# music-mental-health-predictor-random-forest

A machine learning project that explores the relationship between **music listening habits** and **mental health scores** using survey data.  
This project answers the question:  
**Can someone's favorite music genre and daily listening time predict their mental health level?**
**Source:** [MXMH Survey Results on Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)


We created a new target called `Mental_Health_Label`:
- `1`: Higher than median mental health score
- `0`: Lower than median score

This is derived from the average of:
- Anxiety
- Depression
- Insomnia
- OCD

---


We used **Random Forest Classifier** for its robustness and feature interpretability.
