#!/usr/bin/env python
# -*- coding: utf-8 -*-
# usage: python achain.py -n 4 -k 10 -f 京都 -t 東京 --metric alphacos -a 0.2 --strict
from __future__ import print_function
import argparse
import time
import numpy as np
import scipy.spatial.distance as ssd
from collections import deque
from pyknp import Juman
import sys
sys.path.insert(0, "/home/inamura/workspace/gensim")
from gensim.models import Word2Vec


# customize scipy.spatial.distance.cdist in scipy v0.16.1
# for fast computation of weighted cosine distance
def my_cdist(XA, XB, metric='cosine', W=None, alpha=-1):
    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')
    s = XA.shape
    sB = XB.shape
    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')
    mA = s[0]
    mB = sB[0]
    dm = np.zeros((mA, mB), dtype=np.double)

    def _convert_to_double(X):
        if X.dtype != np.double:
            X = X.astype(np.double)
        if not X.flags.contiguous:
            X = X.copy()
        return X

    def _row_norms(X):
        norms = np.einsum('ij,ij->i', X, X)
        return np.sqrt(norms, out=norms)

    def _cosine_cdist(XA, XB, dm):
        XA = _convert_to_double(XA)
        XB = _convert_to_double(XB)
        normsA = _row_norms(XA)
        normsB = _row_norms(XB)
        np.dot(XA, XB.T, out=dm)

        dm /= normsA.reshape(-1, 1)
        dm /= normsB
        dm *= -1
        dm += 1
        return dm

    if isinstance(metric, str):
        mstr = metric.lower()
        if mstr in set(['cosine', 'cos']):
            dm = _cosine_cdist(XA, XB, dm)
        elif mstr in set(['alphacos']) and W is not None and 0 <= alpha <= 1:
            W = np.kron(np.array([[1] for _ in range(mA)]), W)
            dm = (1-alpha) * _cosine_cdist(XA, XB, dm) + \
                alpha * _cosine_cdist(W, XB, dm)
    else:
        raise TypeError('2nd argument metric must be a string identifier ')
    return dm


def argmin_from_V(xi, V, n, metric='cosine', w=None, alpha=-1):
    if metric == 'cosine':
        D = my_cdist([xi], V, metric)[0]
    elif metric == 'alphacos' and w is not None and 0 <= alpha <= 1:
        D = my_cdist([xi], V, metric, W=w, alpha=alpha)[0]
    else:
        print("metric options are wrong")
        exit(1)
    return sorted(enumerate(D), key=lambda x: x[1])[:n]


def get_nearest_words(dists_kv, checked_words, id2vocab, kbest=1, debug=True):
    xs = []
    ds = []
    k = 1
    for i, d in dists_kv:
        tmp_w = id2vocab[i]
        if tmp_w not in checked_words:
            xw = tmp_w
            k += 1
            xs.append(xw)
            ds.append(d)
            if debug:
                print(u"\t{0} not found (d:{1:.3f})".format(tmp_w, d))
            if k > kbest:
                break
        else:
            if debug:
                print(u"\t{0} found (d:{1:.3f})".format(tmp_w, d))
    return xs, ds


def evaluate_n(xws_d_que, fword, tword, topL, metric, model):
    def weighted_average(n, d_rest, d_last):
        return d_rest * 1. / n + d_last * (n-1.) / n
    xwordsd_rslts = []
    n = len(xws_d_que[0][0]) + 1  # number of distances
    print("")
    print("chain length: {}".format(n+1))
    for xws, d in xws_d_que:
        findist = ssd.cosine(model[xws[-1]], model[tword])
        # if metric == 'alphacos' and n==2:
        #     xwordsd_rslts.append((xws, d))
        # else:
        xwordsd_rslts.append((xws, d + findist))
    rslts = sorted(xwordsd_rslts, key=lambda x: x[1])

    dft = ssd.cosine(model[fword], model[tword])
    print(u"results(top-L): from {0} to {1} (d:{2:.3f})"
          .format(fword, tword, dft))
    for i in range(topL):
        xws, d = rslts[i]
        for x in xws:
            print(u'{} '.format(x), end='')
        fd = ssd.cosine(model[fword], model[x])
        td = ssd.cosine(model[tword], model[x])
        print(u'\t d:{0:.3f} (di:{1:.3f}, dt:{2:.3f})'.format(d, fd, td))
        # for evaluation
        prevx = fword
        dsum = 0
        for x in xws:
            di = ssd.cosine(model[prevx], model[x])
            dsum += di
            print(u'{0:.3f} + '.format(di), end='')
            prevx = x
        findist = ssd.cosine(model[prevx], model[tword])
        dsum_rest = dsum
        dsum += findist
        wavg = weighted_average(n, dsum_rest, findist)
        print(u'{0:.3f} = {1:.3f}, (mean: {2:.3f}, 1:{3}-mean: {4:.3f})'.
              format(findist, dsum, dsum/n, n-1, wavg))
        for x in xws:
            fd = ssd.cosine(model[fword], model[x])
            td = ssd.cosine(model[tword], model[x])
            print(u'to_dist:{0:.3f}, from_dist:{1:.3f} | '.format(fd, td), end='')
        print("")
    return rslts[:topL]


def execute(model, fword, tword, V, K, alpha, id2vocab, debug, metric, L, N,
            strict, tprev):
    def append_que(que, p_xws, p_d, n_xws, n_ds, xf, xt, debug, model):
        for xw, d in zip(n_xws, n_ds):
            que.append((p_xws + [xw], p_d + d))
            if debug:
                for x in p_xws + [xw]:
                    print(u'{} '.format(x), end='')
                    df = ssd.cosine(xf, model[xw])
                    dt = ssd.cosine(xt, model[xw])
                    print(u'\t d:{0:.3f} (di:{1:.3f}, dt:{2:.3f})'.
                          format(p_d + d, df, dt))
        return que

    def update_que(que, x, cws, xws, d, consts):
        xf = consts['xf']
        xt = consts['xt']
        V = consts['V']
        id2vocab = consts['id2vocab']
        K = consts['K']
        metric = consts['metric']
        alpha = consts['alpha']
        debug = consts['debug']
        model = consts['model']
        if metric == 'alphacos':
            dists_kv = argmin_from_V(x, V, K+len(cws), metric,
                                     w=xt, alpha=alpha)
        elif metric == 'cosine':
            dists_kv = argmin_from_V(x, V, K+len(cws), metric)
        kbest_xws, kbest_ds = get_nearest_words(dists_kv, cws, id2vocab,
                                                kbest=K, debug=debug)
        que = append_que(que, xws, d, kbest_xws, kbest_ds,
                         xf, xt, debug, model)
        return que

    assert N > 2
    xf = model[fword]
    xt = model[tword]
    # checked_words: 既に見た単語
    checked_words = [fword, tword]
    # Que: 経路の通過履歴(K-best)
    xws_d_que = deque()
    x = xf
    consts = {'xf': xf, 'xt': xt, 'V': V, 'id2vocab': id2vocab, 'model': model,
              'K': K, 'metric': metric, 'alpha': alpha, 'debug': debug}
    xws_d_que = update_que(xws_d_que, x, checked_words, [], 0, consts)
    evaluate_n(xws_d_que, fword, tword, L, metric, model)
    if N == 3:
        return
    for n in range(3, N+1):  # n is chain length (not used)
        # Que: O(K^(n-2)) for each loop
        print("")
        print(".{}".format(len(xws_d_que)))
        for _ in range(len(xws_d_que)):
            print("..{}".format(len(xws_d_que)), end='')
            sys.stdout.flush()
            xws, d = xws_d_que.popleft()
            cws = checked_words + xws
            x = model[xws[-1]]  # x looking at now
            xws_d_que = update_que(xws_d_que, x, cws, xws, d, consts)
        # Que: O(K^(n-2)) -> O(K(n-2))
        if not strict:
            xws_d_que = deque(sorted(xws_d_que, key=lambda x: x[1])[:K])
        evaluate_n(xws_d_que, fword, tword, L, metric, model)

        tnow = time.clock() - tprev
        if debug:
            printtime(tnow)
        tprev = tnow


def initialize(fword, tword, modelfn, start, debug):
    juman = Juman()
    # parse and check from_word
    ms_f = juman.analysis(fword).mrph_list()
    if len(ms_f) > 1:
        print(u'{} is parsed multiple words'.format(fword))
        exit(1)
    wm_f = ms_f[0]
    if not wm_f.repname:
        print(u'no repname with {}'.format(fword))
        exit(1)
    fword = wm_f.repname
    # parse and check to_word
    ms_t = juman.analysis(tword).mrph_list()
    if len(ms_t) > 1:
        print(u'{} is parsed multiple words'.format(tword))
        exit(1)
    wm_t = ms_t[0]
    if not wm_t.repname:
        print(u'no repname with {}'.format(tword))
        exit(1)
    tword = wm_t.repname
    # load and check model
    print(u'loading model...')
    if modelfn.split('.')[-1] == 'model':
        model = Word2Vec.load(modelfn)
    elif modelfn.split('.')[-1] == 'bin':
        model = Word2Vec.load_word2vec_format(modelfn, binary=True, unicode_errors='ignore')
    if fword not in model.vocab:
        raise KeyError(u'{} is not found in the model'.format(fword))
        exit(1)
    elif tword not in model.vocab:
        raise KeyError(u'{} is not found in the model'.format(tword))
        exit(1)
    model.save('hs0.100m.500.5.18mgt100.model')

    t1 = time.clock() - start
    if debug:
        printtime(t1)

    print(u'constructing id2vocab map...')
    id2vocab = {}
    for i, v in enumerate(model.vocab):
        id2vocab[i] = v

    t2 = time.clock() - t1
    if debug:
        printtime(t2)

    print(u'constructing V...')
    V = []
    for v in model.vocab:
        V.append(model[v])
    V = np.vstack(V)

    t3 = time.clock() - t2
    if debug:
        printtime(t3)
    return fword, tword, model, V, id2vocab, t3


def printtime(t):
    print("time:{0:.3f} sec. / {1:.3f} min. / {2:.3f} hours".
          format(t, t/60., t/3600.))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--fword', '-f', type=str, default='京都', help="")
    parser.add_argument('--tword', '-t', type=str, default='東京', help="")
    parser.add_argument('--model', '-m', type=str,
                        default='hs0.100m.500.5.small183313.bin',
                        help="embedding vectors / similarity data")
    parser.add_argument('--num', '-n', type=int, default=2, help="")
    parser.add_argument('--top', type=int, default=10, help="")
    parser.add_argument('--kbest', '-k', type=int, default=100, help="")
    parser.add_argument('--strict', action='store_true', help="")
    parser.set_defaults(debug=False)
    parser.add_argument('--debug', action='store_true', help="")
    parser.set_defaults(debug=False)
    parser.add_argument('--metric', type=str, default='alphacos',
                        help="stage cost metric")
    parser.add_argument('--alpha', '-a', type=float, default=0.5,
                        help="in [0,1]")

    args = parser.parse_args()
    fword = args.fword.decode('utf8')
    tword = args.tword.decode('utf8')
    modelfn = args.model  # '/maple/share7/word2vec/allsim/ja_50000.txt.model'
    N = args.num
    K = args.kbest
    L = args.top
    if N < 3:
        print("N must be >= 3")
        exit(1)
    metric = args.metric
    alpha = args.alpha
    strict = args.strict
    debug = args.debug

    start = time.clock()

    tpl = initialize(fword, tword, modelfn, start, debug)
    fword, tword, model, V, id2vocab, t3 = tpl
    tprev = t3

    print("")
    print(u'start processing')
    execute(model, fword, tword, V, K, alpha, id2vocab, debug, metric, L, N,
            strict, tprev)

    t4 = time.clock() - t3
    if debug:
        printtime(t4)

    print("")
    print("total time:")
    tfinal = time.clock() - start
    printtime(tfinal)


if __name__ == "__main__":
    main()
