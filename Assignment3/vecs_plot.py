import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from word2vec.w2v import Word2Vec
from RandomIndexing.random_indexing import RandomIndexing
import os

def draw_interactive(x, y, text):
    """
    Draw a plot visualizing word vectors with the posibility to hover over a datapoint and see
    a word associating with it
    
    :param      x:     A list of values for the x-axis
    :type       x:     list
    :param      y:     A list of values for the y-axis
    :type       y:     list
    :param      text:  A list of textual values associated with each (x, y) datapoint
    :type       text:  list
    """
    norm = plt.Normalize(1, 4)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots()
    sc = plt.scatter(x, y, c='b', s=100, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        note = "{}".format(" ".join([text[n] for n in ind["ind"]]))
        annot.set_text(note)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding visualization toolkit')
    parser.add_argument('file', type=str, help='A textual file containing word vectors')
    parser.add_argument('-v', '--vector-type', default='w2v', choices=['w2v', 'ri'])
    parser.add_argument('-d', '--decomposition', default='pca', choices=['svd', 'pca'],
                        help='Your favorite decomposition method')
    args = parser.parse_args()

    #
    # YOUR CODE HERE
    #

    if args.vector_type == 'w2v':
        w2v = Word2Vec.load(args.file)
        W = w2v.get_weights()
        words = w2v.get_all_words()
    else:
        ri = RandomIndexing.load(args.file)
        W = ri.get_weights()
        words = ri.get_all_words()

    if args.decomposition == 'pca':
        decomposition = PCA(2)
    else:
        decomposition = TruncatedSVD(2)

    decomposition.fit(W)
    points = decomposition.transform(W)

    draw_interactive(points[:, 0], points[:, 1], words)
