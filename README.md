# Trajectory_Clustering_Challenge

# About

This is trajectory clustering challenge using a sample of taxi trip GPS trajectories.

# Objective

Cluster the provided trajectories baed on their space/time (relative) similarity and detect the outlier trajectories.

# Usage and data formats

This is a sample of T-Drive trajectory dataset that contains a one-week trajectories of 10,357 taxis. The total number of points in this dataset is about 15 million and the total distance of the trajectories reaches 9 million kilometers. You can find the original dataset [here](https://drive.google.com/file/d/1pzaGZaboOdUxsw7l6hhJDdsH8ZqUeZXs/view?usp=sharing).

The pre-processed trajectory sample to be used for this challenge could be found [here](20190425_ProcessedTaxiTrajectories.csv)

The file contains 152010 records with the following fields:
- *trip_id* (id of the trajectory; points with the same ID belong to the same vehicle trajectory; different ID's - different trajectories.
- *time* (date/time stamp of the GPS reading)
- *x* (x-coordinate of the location - logtitude)
- *y* (y-coordinate of the location - latitude)

Only the trajectories having 20 to 60 records over a 2-hour time window (time windows might vary) are included. Trajectories outside the common spacial frame were excluded.

#Proposed approach

We need to deal with the fact that trajectories are not aligned in time and their measurements are not provided at the regular time intervals. For that purpose:

- Introduce uniform timescale: for each trajectory define the first available measurements together with xy-interpolations at the following 20 checkpoints (moments of time): 5,10,15...,100 min from the time of first measurement; for each of the above checkpoints, linear time interpolations to be defined based on the last available measurement before the checkpoint and the first measurement after it; trajectories having no measurements after 100'th min to be discarded; represent each trajectory as a vector of 21 xy-locations corresponding to the first measurement and 20 checkpoints;

- Try Gaussian Mixture clustering of the above vector representations for different numbers of clusters k=3,4,...20; compute average Silhuette score; pick up k which maximizes the average Silhuette;

- For the selected k visualize the clustering - plot the trajectories using different semi-transparent colors depending on the cluster each trajectory belongs to;

- Detect 0.5\%-outliers (trajectories with low 0.5\% likelihood according to the trained Gaussian Mixture model) and visualize them in bold on top.

#Challenge
1. Implement the approach in iPython notebook, commenting the code and results;
2. Discuss the limitations of k-means clustering if used directly instead of Gaussian Mixture in this case;
3. Discuss the limitations of the approach based on clustering the above vectors of trajectory locations revealing similar/anomalous mobility patterns. Propose any alternative approaches you may think of (just discussion, no implementation required). 



