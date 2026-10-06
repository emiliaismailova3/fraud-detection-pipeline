---
title: Card Fraud Risk Scoring
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8080
pinned: false
short_description: LightGBM fraud model on 590k real card payments
---

# Card Fraud Risk Scoring

Interactive demo of a LightGBM fraud-detection model trained on 590,540 real
e-commerce payments (IEEE-CIS dataset). Pick a real transaction from the test
month or edit the fields; the API returns the fraud probability and a review decision.

- Demo page: `/`
- API documentation: `/docs`
- Source code, SQL features, PyTorch comparison: https://github.com/emiliaismailova3/fraud-detection-pipeline
