import numpy as np
from sklearn.ensemble import IsolationForest
from cluster_centers import get_cluster_centers
from sklearn.preprocessing import minmax_scale

class ACAD:

    def __init__(self, anomalies, unlabel, test, classifer, cluster_algo='kmeans', n_clusters='auto',
                 contamination=0, theta=0.4, alpha='auto', beta='auto', return_proba=True,
                 random_state=2020):

        self.anomalies = anomalies
        self.unlabel = unlabel
        self.contamination = contamination
        self.classifer = classifer
        self.test = test
        self.n_clusters = n_clusters
        self.cluster_algo = cluster_algo
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.return_proba = return_proba
        self.random_state = random_state
        self.centers = get_cluster_centers(self.anomalies, self.n_clusters, self.cluster_algo)

    def cal_weighted_score(self):
        dataset = np.r_[self.anomalies, self.unlabel]
        iforest = IsolationForest(n_estimators=100, contamination=self.contamination,
                                  random_state=self.random_state, n_jobs=-1)
        iforest.fit(dataset)
        isolation_score = -iforest.decision_function(dataset)
        isolation_score_scaled = minmax_scale(isolation_score)

        def cal_similarity_score(arr, centers=self.centers):
            min_dist = np.min([np.square(arr - center).sum() for center in centers])
            similarity_score = np.exp(-min_dist / len(arr))
            return similarity_score

        similarity_score = [cal_similarity_score(arr) for arr in dataset]
        similarity_score_scaled = minmax_scale(similarity_score)
        weighted_score = self.theta * isolation_score_scaled + (1 - self.theta) * similarity_score_scaled
        return weighted_score, isolation_score_scaled, similarity_score_scaled

    def determine_trainset(self):
        weighted_score, isolation_score_scaled, similarity_score_scaled = self.cal_weighted_score()
        self.beta_iso = np.percentile(isolation_score_scaled, 75)
        self.beta_ss = np.percentile(similarity_score_scaled, 85)
        ptt_P_NEW_bool = isolation_score_scaled >= self.beta_iso

        ptt_NEW_N_bool = similarity_score_scaled <= self.beta_ss

        ptt_NEW_bool = ptt_P_NEW_bool * ptt_NEW_N_bool
        ptt_NEW_bool = ptt_NEW_bool[50:]  ## need attend
        ptt_NEW = self.unlabel[ptt_NEW_bool]

        center_new = np.mean(ptt_NEW, axis=0)
        center_new = center_new[np.newaxis, :]

        self.anomalies = np.r_[self.anomalies, ptt_NEW]

        self.centers = np.r_[self.centers, center_new]

        weighted_score, isolation_score_scaled, similarity_score_scaled = self.cal_weighted_score()
        weighted_score = weighted_score**10

        min_score, max_score, median_score = [func(weighted_score) for func in (np.min, np.max, np.median)]
        anomalies_score = weighted_score[:len(self.anomalies)]
        unlabel_scores = weighted_score[len(self.anomalies):]
        self.alpha = np.mean(anomalies_score) if self.alpha == 'auto' else self.alpha
        self.beta = median_score if median_score < self.alpha else np.percentile(weighted_score, 45)
        print("alpha:", self.alpha, "beta:", self.beta, "median_score:", median_score)
        assert self.beta < self.alpha, 'beta should be smaller than alpha.'


        rlb_bool = unlabel_scores <= self.beta
        ptt_bool = unlabel_scores >= self.alpha
        rlb_normal, ptt_anomalies = self.unlabel[rlb_bool], self.unlabel[ptt_bool]
        rlb_normal_score, ptt_anomalies_score = unlabel_scores[rlb_bool], unlabel_scores[ptt_bool]
        rlb_normal_weight = (max_score - rlb_normal_score) / (max_score - min_score)
        ptt_anomalies_weight = ptt_anomalies_score / max_score

        anomalies_weight = anomalies_label = np.ones(len(self.anomalies))
        X_train = np.r_[self.anomalies, ptt_anomalies, rlb_normal]
        weights = np.r_[anomalies_weight, ptt_anomalies_weight, rlb_normal_weight]
        y_train = np.r_[anomalies_label, np.ones(len(ptt_anomalies)), np.zeros(len(rlb_normal))].astype(int)
        return X_train, y_train, weights

    def predict(self):
        X_train, y_train, weights = self.determine_trainset()
        clf = self.classifer
        clf.fit(X_train, y_train, sample_weight=weights)
        u_pred1 = clf.predict(self.unlabel)
        test_pred = clf.predict(self.test)
        if self.return_proba:
            u_prob1 = clf.predict_proba(self.unlabel)[:, 1]
            test_prob = clf.predict_proba(self.test)[:, 1]
            return u_pred1, u_prob1, test_pred, test_prob
        else:
            return u_pred1, test_pred

