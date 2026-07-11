# Entity Relationship Diagram

## Project

OptiCrop – Intelligent Crop Recommendation System

## Entities

- User
- Soil Parameters
- Machine Learning Model
- Crop Recommendation
- Dataset

## Relationships

User (1) ------ (M) Soil Parameters

Soil Parameters (M) ------ (1) Machine Learning Model

Dataset (1) ------ (1) Machine Learning Model

Machine Learning Model (1) ------ (M) Crop Recommendation

User (1) ------ (M) Crop Recommendation

## Purpose

The Entity Relationship Diagram illustrates the interaction between the user, soil parameters, machine learning model, crop recommendation, and the Crop Recommendation Dataset (`Crop_recommendation.csv`). It represents how user-provided soil and environmental data are processed by the trained Machine Learning model to generate the most suitable crop recommendation.