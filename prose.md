# Reddit Data Science Project: Comparing Subreddit Similarity

## Table of Contents:
1. [Background](#background)
2. [Problem](#problem)
3. [Related Work](#related-work)
4. [Data collection](#data-collection)
6. [Classification (Problem 1)](#classification)
7. [Visualization (Problem 2)](#visualization-of-differences)
8. [Clustering (Problem 3)](#clustering)
    - [k-means clustering](#k-means-clustering)
    - [Hierarchical clustering](#hierarchical-clustering)
9. [Text Generation with n-gram](#text-generation-with-n-gram)
10. [Future Investigation](#future-investigation)

# Background:
Reddit is a social media platform based on discussion. Users can post to Reddit, with a post title, body text, links, etc. Other users can then comment on the post and reply to other comments as a form of discussion. Users can then upvote or downvote posts and comments in order to influence visibility. 

### Subreddits:
Given that there is a diverse range of interests and thus possible posts, Reddit is divided into distinct subreddits. Users' posts are restricted to one subreddit at a time, and each subreddit has their own list of rules for submission. These subreddits are created and organized by topic, and each one has a distinct name reflecting that topic.
> For example, /r/dating_advice is a subreddit where users ask and give advice on the dating scene.

![](https://i.imgur.com/TLEcwhO.png)

### Posts:
A subreddit can have many thousands of posts. A reddit post (submission) in any given subreddit has a title, upvote score, body text, and comments. In the backend, it also has a submission id.

![](https://i.imgur.com/DyufEdT.png)

# Problem:
This project is structured in a way such that we first attempt to answer a very basic problem, then build off of that problem to more complex problems.

### Problem 1:
The basic problem is: __"Given a post that came from one of two subreddits, is it possible to determine which subreddit it came from based only on text content?"__

For some subreddits, such as /r/dating_advice and /r/techsupport, this task seems very simple. Any human with knowledge of the topics of each subreddit should be able to classify the post. However for others, such as /r/relationships and /r/relationship_advice, even a human may have trouble differentiating posts from those two subreddits.

Additionally, we restrict our input by removing all words that appear in the subreddit title. We do this in an attempt to uncover more subtle information with regards to the text content. This avoids the situation where classification of a post boils down to discovering the subreddit title within the post.

### Problem 2:
A follow-up question is: __"Given two subreddits, what words exactly differentiate their post content?"__

Once we know we can (or cannot) classify posts, we can try to examine what words led to that decision. This provides useful insight into the tendencies of two subreddits, and in the case of two very similar subreddits, we can also determine subtle differences in the topics.

### Problem 3:
Another follow-up question is: __"Given a list of subreddits, can we cluster them by similarity?"__

Once we know we can classify posts from two subreddits, we can try to get a measure of similarity from the results. With that measure of similarity between every subreddit in our list, we can then attempt to cluster them.

### Side-Problem:
A fun side-problem is: __"With a subreddit's n-gram data, can we generate text that appears to be from that subreddit?"__

This is just for fun, not necessarily related too much to the problems above.

# Related work:
In the blog post "Classifying Reddit Posts With Natural Language Processing and Machine Learning" [[1]](#references), author Britt Allen applies supervised machine learning techniques in order to classify posts from /r/menstruation and /r/BabyBumps into the correct subreddit.

Our work would expand on this by not only comparing more than two subreddits, but also visualizing the learned parameters in a way that reveals the most significant differences. In addition, we will be utilizing TFIDF rather than bag-of-words.

In the website "Interactive map of reddit and subreddit similarity calculator" [[2]](#references), the author provides an interactive map of over 40k clustered subreddits. His methods looked at user comment history, and converted user subreddit overlap into a positive pointwise mutual information (PPMI) matrix. He then clustered subreddits using t-Distributed Stochastic Neighbor Embedding (t-SNE). Similarity was scored with cosine similarity.

Our work doesn't focus on user comment overlap, but solely on post title and body text. We attempt to cluster based on our classification results, and we use k-means and hierarchical clustering. Our work is parallel, but with different methods and different implications. While the website doesn't consider content, ours is based fully on text content.

# Data collection:
For a given subreddit, our data consists of top and hot posts. (Most upvoted posts of all time, and "hot" recent posts as designated by reddit's own algorithm).

For a given post, we are interested in the body text and title text.

We collected the initial data using the PRAW API, which accesses reddit through a reddit account. The PRAW API has an intuitive flow, where one can get a `Subreddit` object, then its `Submission`s.

After putting all of the data in a Pandas DataFrame, we save it to file to avoid having to redownload everything.

## Authentication for PRAW API:
This code section has been deleted to avoid revealing our secret key.

## List of Subreddits:
All in all, 33 different subreddits. All of them were hand-picked, with the only criteria being that the majority of the posts had to have substantial text in the title or body. After initial selection, subreddits with similar topics were also added.

## List of Stopwords:
Here, we remove words that exist in the subreddit names, in addition to other non-words and other manually selected stopwords.

## Downloading Submission Data from Internet:
Functions that access the PRAW API and return a Pandas DataFrame with all of our relevant data.

## Caching Submission data by Saving to/Reading from File:

# Classification

For classification, we focus two subreddits at a time. For purposes of analysis and the rest of the project, we focus on Logistic Regression. But we also include implementation for Complement Naive Bayes and K-Nearest Neighbors out of curiousity and to support the validity of Logistic Regression.

To convert text data into a features usable by our models, we converted our examples into a TFIDF matrix.

## Examples/Benchmarks

To see if our methods work as expected, we test on four distinct pairs of subreddits.

- __/r/ProRevenge and /r/pettyrevenge:__  
We expect these posts to be relatively hard to classify, as both are revenge story based subreddits, with the distinction between "Pro" and "petty" being mostly subjective.
- __/r/relationships and /r/relationship_advice:__  
We expect these posts to be relatively even harder to classify, as both are about relationships, but one is supposed to be more of a space for advice focused posts.
- __/r/tifu and /r/nosleep:__  
We expect these posts to be relatively easy to classify, as /r/tifu is about stories where people mess up in funny ways, where /r/nosleep is about scary stories.
- __/r/askscience and /r/explainlikeimfive:__  
We expect these posts to be relatively hard to classify, but not as much as /r/relationships and /r/relationship_advice. This is because although both subreddits are subreddits where users ask questions pertaining to the world, one is focused more on science, where "eli5" is focused on concepts that users wish to have explained to them "as if they were five".

> After training the model, we also have a feature list sorted by magnitude of the weight assigned by the model. This is important for the next section, and as an answer for Problem 2.

## Visualization of differences
Here, we use the weights assigned by the model to each feature (word) as a measure of significance in the classification. We simply display the features in a wordcloud with frequency corresponding directly to the magnitude of the weights assigned.

# Clustering
For Problem 3, we are interested in clustering submissions from different subreddits, as well as clustering subreddits based on similarity. Our data consists of the 33 subreddits listed above.

## k-means clustering

### Clustering submissions

### Clustering subreddits

## Hierarchical clustering

Using our classification model from the Classification section, we aim to get a measure of similarity between all of our distance subreddits. 

Our process is converting test accuracy to a distance/difference measure, which is then the opposite of similarity.

> Gaurav mentioned that in this case, one would usually actually want to use the training accuracy. However, he said we could just settle for test accuracy, as it produced better results. This could be due to the fact that we only have around 2000 submission examples per subreddit, which may not be enough samples.

First, we notice that if our classification model fits our problem well, the accuracy should range between 0.50 to 1.0. We thus scale this range to 0.0 to 1.0, and treat that as the distance values.

We visualize similarity here: 

[//]: # (Visualize distance matrix)

Note this is not particularly meaningful at the moment. Let us identify clusters using hierarchical clustering.

We simply pass the distance matrix into the correct functions, and generate a dendrogram. The x-axis consists of the distinct subreddits, while the y-axis represents the distance.

We print out some clusters that were identified, and sort the rows and columns of our distance matrix into the output order of the dendrogram, which reveals clustering.

[//]: # (Visualize distance matrix)

[//]: # (Todo, cleanup axis labels and such for visuals)

The clusters identified are consistent with human intuition, which suggests that our classifier accuracy is able to model similarity between subreddits well enough.

# Text Generation with n-gram

Here, for the subreddit /r/tifu, we simply construct an n-gram dictionary based on the top and hot submissions, and then generate text by picking randomly from the probability distributions.

This implementation is based off of homework. I tried to use `nltk`'s generation functions, but I found that it actually produced less coherent results.

# Conclusion
For the first problem, we found that the classification accuracy for Logistic Regression is in line with human ability to classify posts into specific subreddits. We were then able to visualize the specific words that distinguished a subreddit from the other.

For the third problem, we INSERT MAYANK CLUSTERING STUFF HERE, and using hierarchical clustering based off of distances calculated using our classification model, we were able to produce clusters of subreddits that are similar. This implies that the accuracy for our classification is able to model similarity or difference well.

## Future Investigation
As an application for classification, we can attempt to not only analyze different subreddits, but also analyze data from a single subreddit over time. For example, we could go through the classification process with /r/news pre-2016 and post-2016, and then see the distinguishing vocabulary.

We can also include comment content in the future. For this project, getting comments for every submission would take too long for the avaliable time we had to work on it, so we cut out analysis including comments.

Finally, we can also do sentiment analysis, such as negative-positive sentiment analysis. An issue with this however, is the inability for libraries such as `nltk`'s vader-lexicon to detect sarcasm, which is abundant in reddit comments.

## References
1. [Classifying Reddit Posts With Natural Language Processing and Machine Learning](https://towardsdatascience.com/classifying-reddit-posts-with-natural-language-processing-and-machine-learning-695f9a576ecb)
2. [Interactive Map of Reddit and Subreddit Similarity Calculator](https://www.shorttails.io/interactive-map-of-reddit-and-subreddit-similarity-calculator/)
6. [PRAW](https://praw.readthedocs.io/en/latest/)
3. [sklearn](https://scikit-learn.org/stable/)
4. [nltk](https://www.nltk.org/)
5. [scipy](https://www.scipy.org/)
