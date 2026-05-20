# DIY_EKF
This is a basic EKF for position prediction in a GPS environment, designed in a simple way so beginners can easily understand and learn from it.

Input datasets taken from https://rpg.ifi.uzh.ch/zurichmavdataset.html website
This is Pixhawk based Flight Control Board, so the onboard datasets will be as 30Hz of GPS and 50Hz of IMU

The main purpose of this basic EKF is, to sync the timestamp based on the frequency of the two different sensor values and filtering out the bias and noise and predict the final next position 


