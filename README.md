# Project4_Comparative-study-of-Support-Vector-Machines-and-Feed-forward-Multilayer-Perceptrons

Project part of the module INM427, Neural Computing, City University, London.

# 1. Motivation

In 2013, the NBA has started using a new statistical analysis technology, called SportsVU (Official NBA release 2013), 
which allows tracking of every single movement a basketball player makes on the court from the very beginning of the game to the very end. 
Basketball is one the greatest big data adopters in the sports industry, as team managers started using big data insights in order
to make adaptations in their game strategy (e.g. the famous Houston Rockets’ ‘Moreyball’ (Partnow, 2016)).

In this paper, SVM and feed-forward multilayer perceptron trained with backpropagation (MLP) algorithms are evaluated and compared.
Their performance is investigated based on speed and accuracy. Our models will be finely tuned via exhaustive grid search 
with cross-validation, over various sets of hyper-parameters. Due to the depth of data and our mutual passion for basketball, 
NBA data from 2015-2016 season is selected.

# 2. Workflow

1. Data gathering and preparation.

NBA 2015-2016 season shot log data, taken from Kaggle (Kaggle.com) is used. This dataset contains 128,069 observations, 
however using the full dataset caused substantial slowdowns in the search for optimal solutions ,
therefore,  first 20,000 instances were used. Data also has 21 features, out of which, 5 continuous were selected, 
numeric variables, that hold time (touch time, shot clock), distance (shot distance, closest defender distance) 
and basketball specific (dribbling - the number of deceiving actions with a ball, before shot attempt) parameters.
Binary variable, indicating whether the shot was made or missed, was used as output (dependent) variable.

Before the analysis, the data was explored and wrangled in Python. Using our general domain knowledge, 
we have noticed outliers in one of our input variables - touch time, with values more than 24sec (limit of time allowance one attack)
or less than 0. Since outliers only accounted for 0.002% of the data,  these observations were removed from our dataset. 
Another input variable, shot clock, had 4.54% of the data missing. Missing values were replaced with median, 
due to its outlier insensitive properties, and speed of imputation. 
The output variable (shot result) had to be converted from a string variable to categorical variable (from ‘made’, ‘missed’ to made = 1 and missed = -1).

2. Feed-forward multilayer perceptron.

Evolved from McCulloch and Pitts neuron (McCulloch and Pitts 1943) multilayer perceptron is now a relatively old 
and established deep learning algorithm, which contains input layer, one or more hidden layers and an output layer. 
Each layer is composed of one or more artificial neurons in parallel. Each layer of neurons is connected to a layer above and assigned with biases. 
Each connection is assigned with weights. Neurons are activated by using either threshold function or a semi-linear function (e.g. Sigmoid function). 
Network trained via backpropagation. (1986, Pineda) It is used in conjunction with an optimisation method (e.g. gradient descent). 
Multilayer perceptron does not hold assumptions about the distribution of the data, it can therefore model any1 non - 
linear function and has the capacity to generalise unseen data to a great accuracy (Gardner and Dorling 1998).

3. SVM

SVM solves the constrained optimization problem, by searching for a hyperplane,
which separates data points with the greatest geometrical/functional margin. Interestingly, 
one of the major drawbacks to SVMs comes from some of its major assets: 
efficiency in the absence of domain knowledge incorporation; and guarantee for a global optimum solution. (Haykin 1998). 
As a result, there is limited control over the number of support vectors selected, and there is no implicit mechanism to account for the prior knowledge. 
Therefore, the main solving algorithm – Sequential Minimal Optimization (SMO)- 
is relatively slow, even when ‘kernel trick’ is applied.

4. Analysis

Analysis with both models was performed in parallel and could be split into 5 steps.
First of all, the data was prepared, as explained in section 2.2 which involved some necessary operations, 
like missing value and outlier treatments.It was conducted in Python 3.5.
Some other transformations, like standardization6 of variables and transposing it into vectors, as well as core analysis, 
were performed in Matlab 2016b. As a part of the data preparation process, it was split into 80%(16 000 observations) 
training data, and 20% (4000 observations) validation data.

The second part, exhaustive grid-search, took the longest due to the vast number of parameter combinations tested. 
Grids of various parameters for both models could be found in table 2. 
A grid search was performed on validation data, with 10 - fold cross-validation. 
This means that each combination of variables was trained on 90% (3600 observations) of validation data, and tested on 10% (400 observations). 
Such methodology, although very computationally costly, modulates overfitting of parameters to the particular subset of data, 
as well as reducing the effect of chance in comparison of parameters.

Once the best combination of parameters was identified, the model was adjusted and the remaining 80% 
of the data was fed into it. Multiple models have been trained and evaluated with all the same parameters, 
however, different sizes of data in the increments of 1000 observations were used. 
This way, learning curves were derived and which allowed for comparison of the effectiveness of our models 
against marginal gain in data quantity.

The fourth step consisted of feeding the training data into the models with the best parameters once again, 
in order to find the scores along with the predicted labels. It was necessary for comparison of accuracy, performed in later steps. 
Models were trained with 10 fold cross-validation. It is important to note that the 10 fold cross –validation exercise was used 
instead of splitting the data into another subset – testing data. 
This is due to the enhanced utilization of data, provided by k –fold cross-validation.

# 3. Conclusion

Overall, MPL took longer to run due to larger number of optimizeable paramters, however, as per iteration, SVM is more computationally demanding.
It was determined that the learning curve for SVM is much smoother. SVM achieves a greater accuracy than MPL, although by a small margin.
Finding the right parameters for MPL can be very time consuming and can take a lot of trial and error, 
as 1,025 different combinations (that ran for almost 7.5 hours) did not give sufficient results and ultimately, 
the optimal MPL parameters had to be found manually (which took an additional 4-5 hours).
It is almost impossible to determine which factors have the most impact on the outcome of the shot, perhaps this study could be replicated 
by using traditional machine learning algorithms (e.g. logistic regression). It would also be interesting to run the study with a different kind of artificial neural network, 
for example, recurrent neural network, which would be tuned to analysing chunks in time, and look for evidence of ‘hot hand’.(streaks of abnormally high shot accuracy)
