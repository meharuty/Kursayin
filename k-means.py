import random

FEATURES = ["x1 Տարիք", "x2 Ճնշում", "x3 Խոլեստերին", "x4 Կրծքավանդակի ցավ", "x5 Շնչահեղձություն"]

DATA = [
    (1, 1, 0, 0, 0), (0, 0, 1, 0, 0), (1, 1, 0, 1, 1), (1, 1, 0, 1, 0), (1, 1, 0, 1, 1),
    (1, 1, 1, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 0, 1), (1, 0, 0, 0, 0), (1, 1, 0, 0, 0),
    (1, 0, 0, 0, 0), (0, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 0, 0, 0, 0),
    (1, 1, 1, 0, 0), (0, 1, 1, 0, 0), (1, 1, 1, 0, 0), (1, 0, 0, 0, 0), (1, 1, 1, 1, 1),
    (0, 0, 1, 0, 0), (0, 1, 0, 1, 1), (1, 0, 1, 0, 0), (1, 1, 0, 1, 0), (0, 0, 0, 0, 0),
    (0, 1, 1, 1, 1), (1, 0, 0, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (0, 0, 0, 0, 0),
    (1, 1, 0, 1, 1), (0, 0, 0, 0, 0), (1, 1, 1, 0, 1), (0, 0, 0, 0, 0), (1, 1, 0, 1, 1),
    (1, 1, 1, 1, 1), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 0, 1, 1), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1),
]


def sq_dist(a, b):
    """Էվկլիդյան հեռավորության քառակուսի"""
    return sum((p - q) ** 2 for p, q in zip(a, b))


class KMeansFromScratch:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = []
        self.labels = []
        self.n_iter = 0

    def assign(self, X):
        """Քայլ 1. յուրաքանչյուր կետ՝ մոտակա կենտրոնին"""
        return [min(range(self.k), key=lambda j: sq_dist(x, self.centroids[j])) for x in X]

    def update(self, X, labels):
        """Քայլ 2. նոր կենտրոն՝ կլաստերի կետերի միջինը"""
        new = []
        for j in range(self.k):
            members = [x for x, l in zip(X, labels) if l == j]
            if members:
                new.append([sum(col) / len(members) for col in zip(*members)])
            else:
                new.append(self.centroids[j])  # դատարկ կլաստեր՝ կենտրոնը մնում է
        return new

    def fit(self, X, init_centroids):
        self.centroids = [list(c) for c in init_centroids]
        for it in range(1, self.max_iter + 1):
            self.labels = self.assign(X)
            new = self.update(X, self.labels)
            self.n_iter = it
            if new == self.centroids:      # կենտրոնները չեն փոխվել՝ զուգամիտություն
                break
            self.centroids = new
        return self

    def inertia(self, X):
        """WCSS = sum ||x - mu_cluster||^2"""
        return sum(sq_dist(x, self.centroids[l]) for x, l in zip(X, self.labels))


def silhouette(X, labels, k):
    """Silhouette գործակցի միջին արժեք"""
    scores = []
    for i, x in enumerate(X):
        own = [sq_dist(x, y) ** 0.5 for j, y in enumerate(X) if labels[j] == labels[i] and j != i]
        a = sum(own) / len(own) if own else 0.0
        b = min(
            sum(sq_dist(x, y) ** 0.5 for j, y in enumerate(X) if labels[j] == c) / labels.count(c)
            for c in range(k) if c != labels[i] and c in labels
        )
        scores.append(0.0 if not own else (b - a) / max(a, b))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    init = [DATA[6], DATA[13], DATA[5]]
    model = KMeansFromScratch(k=3).fit(DATA, init)

    print("Իտերացիաների թիվ:", model.n_iter)
    for j in range(3):
        size = model.labels.count(j)
        print("Կլաստեր %d (%d պացիենտ):" % (j + 1, size),
              [round(v, 3) for v in model.centroids[j]])
    print("Inertia (WCSS):", round(model.inertia(DATA), 3))
    print("Silhouette:", round(silhouette(DATA, model.labels, 3), 3))

    random.seed(5)
    best = min(
        KMeansFromScratch(3).fit(DATA, random.sample(DATA, 3)).inertia(DATA)
        for _ in range(100)
    )
    print("Լավագույն inertia 100 գործարկումից:", round(best, 3))
