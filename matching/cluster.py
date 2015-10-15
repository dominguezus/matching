import itertools


def make_set(x):
    x.parent = x
    x.rank   = 0


def union(x, y):
    xRoot = find(x)
    yRoot = find(y)
    if xRoot.rank > yRoot.rank:
        yRoot.parent = xRoot
    elif xRoot.rank < yRoot.rank:
        xRoot.parent = yRoot
    elif xRoot != yRoot: # Unless x and y are already in same set, merge them
        yRoot.parent = xRoot
        xRoot.rank = xRoot.rank + 1


def find(x):
    if x.parent == x:
        return x
    else:
        x.parent = find(x.parent)
        return x.parent


def cluster(matches):
    matched_entities = []
    for e in matches:
        matched_entities.append(e[0])
        matched_entities.append(e[1])

    for m in matched_entities:
        make_set(m)

    for e in matches:
        union(e[0], e[1])

    grp_key = lambda x:find(x).id
    return {k: set(g) for k,g in itertools.groupby(sorted(matched_entities, key=grp_key), grp_key)}


class cnode(object):
    def __init__(self, x):
        self.id = x
        self.parent = self
        self.rank = 0


def clusterdf(df):
    node_dict = {}
    for m in df.index_x:
        node_dict[m] = cnode(m)

    for idx, e in df.iterrows():
        if e.g1 == 1 and e.g2 == 1:
            union(node_dict[e.index_x], node_dict[e.index_y])

    df['cluster_id'] = df.index_x.apply(lambda x: find(node_dict[x]).id)
    return df
