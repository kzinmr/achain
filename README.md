# achain: fixed length word association generater by word2vec
====

## Description

* 単語の連想系列のようなものを出力するプログラム。系列の長さと両端の単語を与える。
* word2vec は Juman の解析結果に基づいて作成したと仮定している。

<!-- ## Demo -->

<!-- ## VS. -->

## Requirement
* gensim
* word2vec model file
* pyknp (input query parser on which w2v model is base)


## Usage
* python achain.py -n 4 -k 10 -f 京都 -t 東京 -m hs0.100m.500.5.18mgt100.model --metric alphacos -a 0.2 --strict

<!-- ## Install -->

<!-- ## Contribution -->

<!-- ## Licence -->

<!-- [MIT](https://github.com/tcnksm/tool/blob/master/LICENCE) -->

## OPTIONS
* "n" は系列の長さ。3以上のみ許す。
* " -f 京都 -t 東京" は「京都」から始まり、「東京」で終る系列を求めるというオプション。
* "--metric alphacos" は隣接単語を近くする項に、最終単語に近づく項を加えた目的関数。
* "-a 0.2" は目的関数alphacosにおいて、最終単語に近づく度合いの強さ([0,1]内の実数値)。
*  "--strict" は厳密な経路探索を行うオプション。n=4ほどまでなら厳密計算可能。

<!-- ## TODO -->

## Author

[Kazuki Inamura](https://github.com/kzinmr)
