# 💼 Data Science Jobs: Good Salary or Not?

Ever wondered if a data science job pays well just by looking at the company? 🤔 This project tries to guess that!

I cleaned a dataset of data science job posts, trained a **Random Forest** model 🌲, built an API for it with **FastAPI** ⚡, and made a colorful dashboard 📊 where you can try the model and explore the data.

<!-- Add a screenshot: save it as dashboard.png in the project folder and uncomment the next line -->
<!-- ![Dashboard](dashboard.png) -->

## 🎯 What does it do?

- 🔮 **Predicts the salary category** of a job post: **Good Rate** (upper half of salaries) or **Mid to Bad Rate** (lower half), with a probability
- 🎛️ **Filters the data** by company size and seniority level
- 📈 **Shows stats** like how many companies match, the good-rate share, average company size and average post age
- 🥧 **Draws a chart** of which industries have the good-rate companies

## 🛠️ Built with

- 🐍 Python, pandas, scikit-learn
- ⚡ FastAPI
- ⚛️ React and Chart.js (the dashboard is one HTML file, no install needed!)

## 📁 Project files

```
├── jobs_model.py            # 🧹 cleans the data and trains the model
├── jobs_api.py              # ⚡ the FastAPI server
├── jobs_dashboard.html      # 📊 the dashboard
├── data/
│   └── data_science_job_posts_2025.csv   # 📄 the dataset
├── models/                  # 💾 saved model files (made by jobs_model.py)
├── requirements.txt
└── README.md
```

> Change the names above if your folders look different 😉

## 🚀 How to run it

**1. Clone the project** 📥

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Install the requirements** 📦

```bash
pip install -r requirements.txt
```

**3. Train the model** 🏋️ (this creates the files in the `models` folder)

```bash
python jobs_model.py
```

**4. Start the API** 🔌

```bash
uvicorn jobs_api:app --reload
```

**5. Open the dashboard** 🎉

Double-click `jobs_dashboard.html` to open it in your browser. If the top right says **API connected** ✅, everything works! If it says **API offline** ❌, check that the server from step 4 is running.

## 🧠 About the model

- 🌲 **Algorithm:** Random Forest Classifier
- 🧹 **Cleaning:** turned the salary ranges into one number (the middle of the range), changed "3 months ago" style dates into days, filled the missing values, and one-hot encoded the text columns
- 🎯 **Target:** I split the salaries in two equal halves at the median, so `1` = high salary and `0` = low salary
- ✂️ **Split:** 82% training and 18% testing
- 🏆 **Accuracy:** XX% <!-- put your real result here -->

## 📄 Dataset

A data science job posts 2025 dataset with about 940 job posts. It has the job title, seniority level, work status (remote, hybrid, on-site), industry, company size, ownership and salary. <!-- add the source link here -->

## 💡 What I learned

- 🧼 Cleaning messy real-world data (dates, salaries and text columns are never clean!)
- 🎯 Turning a number (salary) into a classification target
- 🤖 Training and saving a model with scikit-learn
- 🔌 Building an API that serves a model
- 🎨 Connecting a frontend to an API

## 🔮 Ideas for the future

- 🛠️ Use the skills column, which I skipped because it was hard to use
- 🧪 Try other models and compare them
- 📊 Show which features matter the most
- 🌍 Put the project online
